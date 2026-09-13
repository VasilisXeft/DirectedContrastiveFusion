import itertools
import torch
from torch import nn
import torch.nn.functional as F
from cmf.models.encoders import TimeSeriesEncoder, ImageSequenceEncoder


class ContrastiveSparseFusion(nn.Module):
    """Directed sparse cross-modal fusion.

    Edge (src, tgt) means information flows src -> tgt. The attention query is the
    target token sequence and key/value are the source tokens. In trainable modes,
    selection is sample-dependent and learned end-to-end with a straight-through
    Gumbel/softmax top-k estimator.
    """
    def __init__(self, modality_shapes, num_outputs, task="classification", d_model=128, heads=4,
                 topk=1, mode="contrastive_topk", temperature=0.1, reliability=True,
                 selector_temperature=0.7, gumbel=True):
        super().__init__()
        self.names = list(modality_shapes)
        self.mode = mode
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.selector_temperature = float(selector_temperature)
        self.use_gumbel = bool(gumbel)
        self.task = task

        self.encoders = nn.ModuleDict()
        for name, shape in modality_shapes.items():
            if len(shape) == 4:  # T,C,H,W
                self.encoders[name] = ImageSequenceEncoder(shape[1], d_model)
            else:
                self.encoders[name] = TimeSeriesEncoder(shape[-1], d_model)

        # Contrastive representation heads.
        self.projectors = nn.ModuleDict({
            n: nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            for n in self.names
        })
        # Separate source/target projections make score(src->tgt) != score(tgt->src).
        self.src_selector = nn.ModuleDict({n: nn.Linear(d_model, d_model, bias=False) for n in self.names})
        self.tgt_selector = nn.ModuleDict({n: nn.Linear(d_model, d_model, bias=False) for n in self.names})

        self.reliability = nn.ModuleDict({n: nn.Linear(d_model, 1) for n in self.names}) if reliability else None

        # Module name src__tgt means src -> tgt.
        self.attn = nn.ModuleDict({
            f"{src}__{tgt}": nn.MultiheadAttention(d_model, heads, batch_first=True)
            for src, tgt in itertools.permutations(self.names, 2)
        })
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(.2), nn.Linear(d_model, num_outputs)
        )

    def _encode(self, modalities):
        toks = {n: self.encoders[n](modalities[n]) for n in self.names}
        pooled = {n: t.mean(1) for n, t in toks.items()}
        z = {n: F.normalize(self.projectors[n](pooled[n]), dim=-1) for n in self.names}
        return toks, pooled, z

    def contrastive_loss(self, z, present):
        # Symmetric InfoNCE aligns modalities at representation level; selection itself
        # is additionally optimized by downstream task gradients through ST top-k.
        losses = []
        names = self.names
        for i, a in enumerate(names):
            for b in names[i+1:]:
                valid = (present[a] > 0) & (present[b] > 0)
                if valid.sum() > 1:
                    idx = torch.where(valid)[0]
                    za, zb = z[a][idx], z[b][idx]
                    logits = za @ zb.T / self.temperature
                    lab = torch.arange(len(idx), device=logits.device)
                    losses += [F.cross_entropy(logits, lab), F.cross_entropy(logits.T, lab)]
        return torch.stack(losses).mean() if losses else next(iter(z.values())).new_tensor(0.)

    def _reliability(self, pooled):
        bsz = len(next(iter(pooled.values())))
        dev = next(iter(pooled.values())).device
        return {
            n: torch.sigmoid(self.reliability[n](pooled[n])).squeeze(-1)
            if self.reliability is not None else torch.ones(bsz, device=dev)
            for n in self.names
        }

    def _directional_scores(self, z, present, rel):
        scores = {}
        for src, tgt in itertools.permutations(self.names, 2):
            qs = F.normalize(self.src_selector[src](z[src]), dim=-1)
            kt = F.normalize(self.tgt_selector[tgt](z[tgt]), dim=-1)
            s = (qs * kt).sum(-1)
            # Source reliability matters most for outgoing information; target presence
            # still gates the edge. A mild target factor stabilizes missing-modality cases.
            s = s + torch.log(rel[src].clamp_min(1e-4)) + 0.25 * torch.log(rel[tgt].clamp_min(1e-4))
            valid = present[src] * present[tgt]
            s = torch.where(valid > 0, s, torch.full_like(s, -1e4))
            scores[(src, tgt)] = s
        return scores

    def _static_similarity_scores(self, z, present):
        scores = {}
        for src, tgt in itertools.permutations(self.names, 2):
            s = (z[src] * z[tgt]).sum(-1)
            valid = present[src] * present[tgt]
            scores[(src, tgt)] = torch.where(valid > 0, s, torch.full_like(s, -1e4))
        return scores

    def _st_topk(self, mat):
        # mat: [B, num_targets]. Forward pass is hard top-k; backward uses softmax.
        k = min(self.topk, mat.shape[1])
        logits = mat / max(self.selector_temperature, 1e-5)
        if self.training and self.use_gumbel:
            u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
            logits = logits - torch.log(-torch.log(u))
        probs = F.softmax(logits, dim=1)
        idx = probs.topk(k, dim=1).indices
        hard = torch.zeros_like(probs).scatter_(1, idx, 1.0)
        if self.training:
            return hard + probs - probs.detach(), hard
        return hard, hard

    def _selection(self, scores, batch_size, device):
        weights = {p: torch.zeros(batch_size, device=device) for p in scores}
        hard_masks = {p: torch.zeros(batch_size, dtype=torch.bool, device=device) for p in scores}

        if self.mode == "full":
            for p in weights:
                weights[p].fill_(1.0); hard_masks[p].fill_(True)
        elif self.mode == "random_topk":
            for src in self.names:
                tgts = [t for t in self.names if t != src]
                for bi in range(batch_size):
                    perm = torch.randperm(len(tgts), device=device)[:min(self.topk, len(tgts))]
                    for j in perm.tolist():
                        weights[(src, tgts[j])][bi] = 1.0
                        hard_masks[(src, tgts[j])][bi] = True
        elif self.mode in {"contrastive_topk", "directed_topk", "similarity_topk"}:
            for src in self.names:
                tgts = [t for t in self.names if t != src]
                mat = torch.stack([scores[(src, t)] for t in tgts], dim=1)
                if self.mode in {"contrastive_topk", "directed_topk"}:
                    st, hard = self._st_topk(mat)
                else:
                    idx = mat.topk(min(self.topk, len(tgts)), dim=1).indices
                    hard = torch.zeros_like(mat).scatter_(1, idx, 1.0)
                    st = hard
                for j, tgt in enumerate(tgts):
                    weights[(src, tgt)] = st[:, j]
                    hard_masks[(src, tgt)] = hard[:, j].bool()
        elif self.mode == "late":
            pass
        else:
            raise ValueError(f"Unknown fusion mode {self.mode}")
        return weights, hard_masks

    def forward(self, modalities, present):
        toks, pooled, z = self._encode(modalities)
        rel = self._reliability(pooled)
        scores = (self._static_similarity_scores(z, present)
                  if self.mode == "similarity_topk"
                  else self._directional_scores(z, present, rel))

        bsz = next(iter(pooled.values())).shape[0]
        dev = next(iter(pooled.values())).device
        weights, hard_masks = self._selection(scores, bsz, dev)

        enriched = {n: pooled[n] * rel[n].unsqueeze(-1) * present[n].unsqueeze(-1) for n in self.names}
        counts = {n: present[n].clone() for n in self.names}
        pair_count = torch.zeros(bsz, device=dev)

        if self.mode != "late":
            for (src, tgt), w in weights.items():
                if not hard_masks[(src, tgt)].any() and not (self.training and w.requires_grad):
                    continue
                # src -> tgt: target queries source; result enriches target representation.
                y, _ = self.attn[f"{src}__{tgt}"](toks[tgt], toks[src], toks[src], need_weights=False)
                y = y.mean(1)
                enriched[tgt] = enriched[tgt] + y * w.unsqueeze(-1)
                counts[tgt] = counts[tgt] + w
                pair_count += hard_masks[(src,tgt)].float()

        per = [enriched[n] / counts[n].clamp_min(1).unsqueeze(-1) for n in self.names]
        stack = torch.stack(per, dim=1)
        pmask = torch.stack([present[n] for n in self.names], dim=1).unsqueeze(-1)
        fused = (stack * pmask).sum(1) / pmask.sum(1).clamp_min(1.)
        out = self.head(self.norm(fused))

        aux = {
            "contrastive_loss": self.contrastive_loss(z, present),
            "pair_count": pair_count.mean(),
            "reliability": rel,
            "scores": scores,
            "hard_masks": hard_masks,
        }
        return out, aux
