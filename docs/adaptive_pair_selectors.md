# Adaptive pair-selection benchmark (experimental)

The existing `ContrastiveSparseFusion` now accepts four additional modes:
`hsic_topk`, `clam_topk`, `utility_topk`, and `hybrid_topk`.
All use the same cached DEAP features, encoders, fusion head, and LOSO split.
The `directed_topk`, `contrastive_topk`, and `similarity_topk` baselines remain unchanged.

## Run a single held-out subject

```powershell
git checkout feature/adaptive-pair-selectors
pip install -e .
pytest -q tests/test_pair_selectors.py
python scripts/run_deap_external_5m.py --config configs/deap_external_5m.yaml --mode hsic_topk --topk 10 --subject s22 --seed 42
python scripts/run_deap_external_5m.py --config configs/deap_external_5m.yaml --mode clam_topk --topk 10 --subject s22 --seed 42
python scripts/run_deap_external_5m.py --config configs/deap_external_5m.yaml --mode utility_topk --topk 10 --subject s22 --seed 42
python scripts/run_deap_external_5m.py --config configs/deap_external_5m.yaml --mode hybrid_topk --topk 10 --subject s22 --seed 42
```

Omit `--subject` for all LOSO folds. Keep `--topk`, seed, and data splits fixed across selectors.
For a fair trial-level comparison, aggregate window predictions from all variants using the
same trial-level rule; these commands use the original window-level classifier.

## What the modes actually measure

* `hsic_topk`: centered normalized Gram alignment across **tokens within one window**.
  It is a *within-window HSIC-inspired dependence proxy*, not population HSIC.
  With fewer than three tokens, the score is zero; for a one-token PPG embedding,
  this selector cannot establish dependency and may under-select PPG. Scores are symmetric
  across source/target and do **not** establish directed dependence.
* `clam_topk`: learned gated-like attention logits over ordered source-target pair features
  plus attention-rank pseudo-label clustering. This is CLAM-inspired, **not original CLAM**,
  and clustering labels are self-generated rather than class-specific.
* `utility_topk`: predicts a detached leave-one-selected-edge-out classification-loss
  delta. It only supervises edges selected by the current policy, and the teacher is
  based on the model's current predictions and labels; it is not a causal estimate.
  Extra forward passes through the classification head add training overhead.
* `hybrid_topk`: weighted sum of token-dependence proxy, attention logits, and predicted
  utility. These terms have different scales; tune weights using validation subjects only.

All learned selectors use straight-through Top-K. Non-selected attention branches may
still be evaluated during training; **Top-K is an interaction budget, not a FLOPs guarantee**.
The quality penalty is computed from cached representations, not raw physiological SQI.

## Recommended protocol

First run `pytest` and one subject as a technical smoke test, then use a separate
validation subject for hyperparameter selection, and evaluate on held-out test subjects.
Inspect edge-selection frequencies, reliability, per-subject macro-F1, and variability
over multiple seeds. Compare against `directed_topk`, `contrastive_topk`,
`similarity_topk`, `random_topk`, `late`, and `full`.

These implementations have **not been trained or validated against the user's DEAP cache**.
