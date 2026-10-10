"""Attention-based trial-level MIL over the existing directed fusion model."""
import torch
from torch import nn
from cmf.models.fusion import ContrastiveSparseFusion


class DirectedFusionMIL(nn.Module):
    def __init__(self, modality_shapes, num_classes, model_cfg):
        super().__init__()
        cfg = model_cfg
        d = int(cfg.get("d_model", 128))
        self.fusion = ContrastiveSparseFusion(
            modality_shapes, num_classes, task="classification", d_model=d,
            heads=cfg.get("heads", 4), topk=cfg.get("topk", 10),
            mode=cfg.get("mode", "directed_topk"),
            temperature=cfg.get("temperature", 0.1),
            reliability=cfg.get("reliability", True),
            selector_temperature=cfg.get("selector_temperature", 0.7),
            gumbel=cfg.get("gumbel", True),
            encoder_configs=cfg.get("encoders", {}),
            quality_lambda=cfg.get("quality_lambda", 1.0),
            reliability_floor=cfg.get("reliability_floor", 0.25),
            reliability_src_weight=cfg.get("reliability_src_weight", 0.5),
            reliability_tgt_weight=cfg.get("reliability_tgt_weight", 0.25))
        self.attention = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.classifier = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                        nn.Dropout(0.2), nn.Linear(d, num_classes))

    def forward(self, modalities, present, quality_penalty, mask):
        # Compute fusion only on real windows, excluding padded instances entirely.
        flat = mask.reshape(-1)
        b, t = mask.shape
        mods = {n: x.reshape(b*t, *x.shape[2:])[flat] for n, x in modalities.items()}
        pres = {n: x.reshape(b*t)[flat] for n, x in present.items()}
        qual = {n: x.reshape(b*t)[flat] for n, x in quality_penalty.items()}
        _, aux = self.fusion(mods, pres, qual)
        d = aux["fused_embedding"].shape[-1]
        features = aux["fused_embedding"].new_zeros((b*t, d))
        features[flat] = aux["fused_embedding"]
        features = features.reshape(b, t, d)
        scores = self.attention(features).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        bag_embedding = (features * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(bag_embedding)
        return logits, {"attention": weights, "pair_count": aux["pair_count"],
                        "reliability": aux["reliability"]}
