"""Selector smoke tests; run with pytest -q tests/test_pair_selectors.py."""
import pytest
import torch
from cmf.models.fusion import ContrastiveSparseFusion


@pytest.mark.parametrize("mode", ["hsic_topk", "clam_topk", "utility_topk", "hybrid_topk"])
def test_selector_train_and_eval(mode):
    torch.manual_seed(5)
    shapes = {"eeg": (4, 8), "eda": (4, 8), "face": (4, 8)}
    model = ContrastiveSparseFusion(
        shapes, 2, d_model=16, heads=4, topk=2, mode=mode,
        encoder_configs={n: {"type": "precomputed", "feature_dim": 8} for n in shapes},
        gumbel=False)
    modalities = {n: torch.randn(3, *shape) for n, shape in shapes.items()}
    present = {n: torch.ones(3) for n in shapes}
    quality = {n: torch.zeros(3) for n in shapes}
    labels = torch.tensor([0, 1, 0])
    model.train()
    logits, aux = model(modalities, present, quality, targets=labels)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux["selector_loss"])
    assert abs(float(aux["pair_count"]) - 2) < 1e-5
    loss = torch.nn.functional.cross_entropy(logits, labels) + 0.1*aux["selector_loss"]
    loss.backward()
    assert any(p.grad is not None for p in model.pair_selector.parameters())
    model.eval()
    with torch.no_grad():
        pred, diag = model(modalities, present, quality)
    assert torch.isfinite(pred).all()
    assert len(diag["scores"]) == 6


def test_missing_modality_not_selected():
    shapes = {"a": (4, 8), "b": (4, 8), "c": (4, 8)}
    model = ContrastiveSparseFusion(shapes, 2, d_model=16, heads=4,
                                   mode="clam_topk", topk=2,
                                   encoder_configs={n: {"type": "precomputed", "feature_dim": 8} for n in shapes})
    mods = {n: torch.randn(2, 4, 8) for n in shapes}
    present = {n: torch.ones(2) for n in shapes}
    present["c"][:] = 0
    model.eval()
    with torch.no_grad():
        _, aux = model(mods, present)
    for (src, tgt), selected in aux["hard_masks"].items():
        if "c" in (src, tgt):
            assert not selected.any()
