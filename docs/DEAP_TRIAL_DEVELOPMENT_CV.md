# Trial-level DEAP development validation

## Pilot results: October 12, 2026

Each trial is one labeled 60s DEAP stimulus. Each modality contributes 11
chronological tokens pooled from the cached windows. Scaling is fit on
18 training subjects; validation on s07/s14/s21; reserved test s22 is untouched.

| Mode | Subject mean balanced accuracy | Subject mean macro-F1 |
| --- | ---: | ---: |
| Ridge EDA | .5786 | .5655 |
| Ridge EEG | .5597 | .5321 |
| Ridge concat | .5866 | .5010 |
| Neural EEG | .5923 | .5819 |
| Neural EDA | .5592 | .5504 |
| Early fusion | .6081 | .5188 |
| Directed top-k=5 | .5916 | .5178 |

Models used seed 42, a 15 epoch maximum, 5 early-stopping patience,
8 trials per batch and training-fitted modality feature normalization.
Directed fusion did not outperform EEG-only on these development metrics.
Its training loss fell 0.697 -> 0.295 while validation loss increased
0.691 -> 1.276 by epoch 10, indicating serious overfitting.

The pilot used the same 3 subjects both to pick checkpoints and to
measure performance. These are NOT unbiased held-out test estimates.

## Seven-group development validation

The 21 development participants are partitioned into seven disjoint
3-subject validation groups. Each fold trains on 18 subjects and never
loads or evaluates reserved s22:

- Group 0: s01, s08, s15
- Group 1: s02, s09, s16
- Group 2: s03, s10, s17
- Group 3: s04, s11, s18
- Group 4: s05, s12, s19
- Group 5: s06, s13, s20
- Group 6: s07, s14, s21

Inspect folds without training:

    python scripts/run_deap_trial_dev_cv.py --dry-run

First evaluate simple baselines in all groups with a fixed configuration:

    python scripts/run_deap_trial_dev_cv.py --modes ridge_all unimodal_eeg early --seeds 42 --epochs 15 --patience 5 --batch-size 8

Then evaluate the directed model on the same groups:

    python scripts/run_deap_trial_dev_cv.py --modes directed_topk --seeds 42 --epochs 15 --patience 5 --batch-size 8

Then compare all models from disk without retraining:

    python scripts/run_deap_trial_dev_cv.py --modes ridge_all unimodal_eeg early directed_topk --seeds 42 --aggregate-only

Additional seeds 43/44 can be included after smoke tests.
Every group refits normalization exclusively on its training subjects.
Runs are resumable; output is separated by validation group and seed.

The runner saves summary.csv, fold_metrics.csv and subject_metrics.csv
including one-class prediction counts. Each model also stores its best
checkpoint and a detailed per-subject validation metrics JSON.

## Additional baselines still missing from the initial pilot

To compare late and full fusion without changing the original settings:

    python scripts/run_deap_trial_fusion.py --modes late full --epochs 15 --patience 5 --batch-size 8 --out runs/deap_foundation_5m/trial_fusion_pilot --skip-existing

## Scientific cautions

The held-out development validation participants are also used for
checkpoint selection, so validation results remain potentially optimistic.
For publication, freeze hyperparameters prior to final testing or use
nested subject-independent evaluation. Never tune against s22.

The straight-through directed selector uses a soft-gradient path, which
may still compute attention for ALL 20 directed pairs during training
even when the reported hard pair count is only 5. Sparse edge count is
not proof of lower training FLOPs.

Large cross-attention parameter counts versus only 715 independently
labeled training trials plausibly explain the overfitting. If the
seven-group results remain weak, evaluate smaller embedding dimensions
and stronger regularization in matched, development-only ablations.
