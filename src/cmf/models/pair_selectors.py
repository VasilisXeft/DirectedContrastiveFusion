"""Pluggable directed pair scores.

HSIC uses within-sample token Gram alignment (a descriptive dependence proxy,
not a directional/causal estimator). CLAM-inspired uses gated attention over
ordered pair features. Utility-guided learns a surrogate for edge ablation
loss deltas; its teacher is supplied separately by the fusion model.
"""
import itertools
import torch
from torch import nn
import torch.nn.functional as F


class PairSelector(nn.Module):
    MODES = {"hsic_topk", "clam_topk", "utility_topk", "hybrid_topk"}

    def __init__(self, names, d_model, mode, hidden=128, utility_weight=1.0,
                 attention_weight=1.0, hsic_weight=1.0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unknown selector mode %s" % mode)
        self.names = list(names)
        self.pairs = list(itertools.permutations(names, 2))
        self.mode = mode
        self.utility_weight = float(utility_weight)
        self.attention_weight = float(attention_weight)
        self.hsic_weight = float(hsic_weight)
        # Ordered concatenation makes i->j distinct from j->i.
        self.pair_features = nn.ModuleDict({
            "%s__%s" % p: nn.Sequential(nn.Linear(4*d_model, hidden),
                                         nn.GELU(), nn.LayerNorm(hidden))
            for p in self.pairs})
        self.attention = nn.Sequential(nn.Linear(hidden, hidden),
                                       nn.Tanh(), nn.Linear(hidden, 1))
        self.utility = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, 1))
        # Auxiliary CLAM-inspired discriminative clustering of pair representations.
        self.cluster = nn.Linear(hidden, 2)

    @staticmethod
    def _hsic_proxy(a, b):
        # Tokens within ONE window are observations. Equal token counts are
        # needed for alignment; interpolate along token axis if necessary.
        n = min(a.shape[1], b.shape[1])
        if n < 3:
            return a.new_zeros(a.shape[0])
        a = F.adaptive_avg_pool1d(a.transpose(1, 2), n).transpose(1, 2)
        b = F.adaptive_avg_pool1d(b.transpose(1, 2), n).transpose(1, 2)
        a = F.normalize(a-a.mean(1, keepdim=True), dim=-1)
        b = F.normalize(b-b.mean(1, keepdim=True), dim=-1)
        ka = a @ a.transpose(1, 2)
        kb = b @ b.transpose(1, 2)
        ka = ka-ka.mean(1, keepdim=True)-ka.mean(2, keepdim=True)+ka.mean((1, 2), keepdim=True)
        kb = kb-kb.mean(1, keepdim=True)-kb.mean(2, keepdim=True)+kb.mean((1, 2), keepdim=True)
        return (ka*kb).sum((1, 2))/(
            ka.square().sum((1, 2)).sqrt()*kb.square().sum((1, 2)).sqrt()+1e-6)

    def forward(self, z, toks, present, rel):
        representations = []
        hsic = []
        for src, tgt in self.pairs:
            x = torch.cat([z[src], z[tgt], z[src]*z[tgt], z[src]-z[tgt]], dim=-1)
            representations.append(self.pair_features["%s__%s" % (src, tgt)](x))
            hsic.append(self._hsic_proxy(toks[src], toks[tgt]))
        h = torch.stack(representations, 1)
        valid = torch.stack([(present[s]>0)&(present[t]>0) for s, t in self.pairs], 1)
        hs = torch.stack(hsic, 1).detach()
        attn_logits = self.attention(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~valid, -1e4)
        # Normalize over valid directed edges only.
        attention = F.softmax(attn_logits, dim=1)*valid.float()
        attention = attention/attention.sum(1, keepdim=True).clamp_min(1e-8)
        utility = self.utility(h).squeeze(-1)
        if self.mode == "hsic_topk":
            score = hs
        elif self.mode == "clam_topk":
            score = attn_logits
        elif self.mode == "utility_topk":
            score = utility
        else:
            # HSIC is symmetric. The other terms provide directionality.
            score = (self.hsic_weight*hs+self.attention_weight*attn_logits+
                     self.utility_weight*utility)
        score = score.masked_fill(~valid, -1e4)
        return {p: score[:, i] for i, p in enumerate(self.pairs)}, {
            "pair_features": h, "pair_attention": attention,
            "utility_predictions": utility, "hsic_proxy": hs,
            "pair_valid": valid, "cluster_logits": self.cluster(h)}

    def cluster_loss(self, aux, topk):
        # Pseudo-instance labels from detached attention rank; CLAM-inspired,
        # not the original CLAM class-specific clustering algorithm.
        valid = aux["pair_valid"]
        scores = aux["pair_attention"].detach().masked_fill(~valid, -1)
        k = min(int(topk), scores.shape[1])
        top = scores.topk(k, dim=1).indices
        targets = torch.zeros_like(scores, dtype=torch.long).scatter_(1, top, 1)
        return F.cross_entropy(aux["cluster_logits"][valid], targets[valid]) if valid.any() else scores.sum()*0
