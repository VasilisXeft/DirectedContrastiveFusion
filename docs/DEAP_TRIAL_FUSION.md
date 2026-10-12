# Normalized, trial-level five-modality DEAP benchmark

## Motivation

The original experiments treated every 10-second DEAP window as an
independently labeled example even though DEAP gives **one valence label
per 60-second stimulus/trial**. Window-level metrics are therefore less
interpretable than trial-level metrics. The raw EEGPT (256Hz, 1e-6)
embeddings reached mean subject-held-out balanced accuracy 0.5277
in 21 development folds; 250Hz, 1e-6 reached 0.4990; 250Hz, 1e-3
reached 0.5200 (paired fixed-ridge tests). EEGPT preprocessing alone
does not explain the entire generalization failure.

We now evaluate all five modalities on true 60s trials using the EXISTING
cache. No new foundation inference is needed for this experiment.

## Trial construction

Each of the 874 valid DEAP trials is represented by **one** example,
consisting of 11 chronologically ordered overlapping 10-second windows
(starts 0,5,...,50 seconds). Every window in the frozen cache has 4
foundation embedding tokens. The script normalizes each token using
train-only per-modality statistics, then averages those 4 tokens into
one vector per 10-second window. The resulting per-trial data are:

- EEG: [11, 512]
- PPG: [11, 1024]
- EDA: [11, 1024]
- TEMP: [11, 1024]
- FACE: [11, 384]
- Target: exactly one binary trial valence label.

No duplicate trials, missing windows or conflicting window labels are
allowed. The original NPZ files stay unchanged.

## Subject-independent development split

On the available 22-subject 5-modality cohort:

- Training: 18 subjects, **715** independently labeled trials
  (**7,865** cached 10-second windows).
- Validation: s21, s14, s07, **119** trials (**1,309** cached windows).
- Reserved final subject: s22, 40 trials. The new development
  benchmark **does not open or evaluate the reserved subject**.

Subjects are partitioned without using outcome labels, using the same
spaced-previous selection as the configured 18/3/1 benchmark. This
is a **development comparison** only: checkpoint/architecture selection
on those three subjects makes their reported metrics optimistic for
model selection. Once the protocol is finalized, evaluate s22 exactly
once. For publication-ready subject-independent generalization,
prefer nested evaluation across outer held-out subjects.

## Normalization (all modalities)

In each development run, the existing FeatureStandardizer is fitted
on **training-window tokens only** and independently for each
modality and feature dimension. The standardization uses the train
mean and standard deviation, minimum std 1e-5, and clips z-scores to
[-8,8] for stability. A matching frozen transform is applied to all
validation tokens; the same fitted tensors are saved with the
checkpoint. No validation/test statistics are used to fit scales.

The '--no-normalization' flag enables a separate controlled ablation,
using a DIFFERENT output directory.

## Benchmarks

Baselines:
- ridge_eeg / ridge_ppg / ridge_eda / ridge_temp / ridge_face:
  train-only standardized trial-mean embeddings, RidgeClassifier alpha 100.
- ridge_concat: concatenation of all five trial-mean vectors, same
  train-only standardized RidgeClassifier.
- unimodal_eeg / unimodal_ppg / unimodal_eda / unimodal_temp /
  unimodal_face: neural single-modality baselines.
- late: feature-level averaging of encoded modalities (NOT
  independent decision-level late fusion).
- early: concatenated pooled representations plus trainable MLP.
- full: all **20 directed cross-modal attention pairs**.
- directed_topk: trainable global directed top-5 selection of 20 pairs
  (use '--topk 10' for a 50% pair budget ablation).
  IMPORTANT: this is an interaction-CONTRIBUTION budget, not a verified
  reduction in training FLOPs. Straight-through selection still computes
  nonselected attention edges during training to propagate gradients;
  inference skips entirely unselected pairs in a batch.

All neural variants use the same backbone cache, the same 18/3
train/validation subjects, the same independent per-feature
normalization, classifier dimensionality (128), loss (class-weighted
cross-entropy), seed, optimizer and training schedule.
Reliability/quality gating and contrastive auxiliary loss are
**disabled on purpose in this stage** to isolate representation
fusion. They should be reintroduced only after establishing this
reliable baseline. The directed-topk selection still learns
directed scores. The full/early/late neural variants use shared
trainable foundation-to-128 projections but learn different fusion.

**Checkpoint selection**: maximize arithmetic mean of **per-subject
macro-F1** across the 3 validation subjects; break ties using mean
per-subject balanced accuracy, and stop after patience=8 epochs without
improvement. Balanced accuracy and macro-F1 are reported separately
for each validation subject, together with pooled metrics, predicted
class counts and confusion matrices. Early stopping does NOT use s22.

## Run

From repo root, after activating the venv:

    git pull origin feature/adaptive-pair-selectors

    python -m unittest discover -s tests -p "test_deap_trial_fusion.py" -v

    python scripts/run_deap_trial_fusion.py --dry-run

Start with a quick no-GPU/low-cost baseline:

    python scripts/run_deap_trial_fusion.py --modes ridge_all

Then a limited comparison using one seed to confirm stable operation:

    python scripts/run_deap_trial_fusion.py --modes unimodal_eeg unimodal_eda early directed_topk --epochs 15 --patience 5 --batch-size 8 --out runs/deap_foundation_5m/trial_fusion_pilot

For the full development comparison, use a SEPARATE output folder:

    python scripts/run_deap_trial_fusion.py --modes ridge_all unimodal_eeg unimodal_ppg unimodal_eda unimodal_temp unimodal_face late early full directed_topk --epochs 30 --patience 8 --batch-size 16 --seed 42 --out runs/deap_foundation_5m/trial_fusion_full

The command reads existing frozen features and computes trial-level
features in memory once. It does not run EEGPT, MOMENT or DINOv2.
On an 8-GB GPU, prefer '--batch-size 8' if full or directed attention
runs out of memory. The script does not silently lower batch size
or fabricate missing results.

## Output

The script writes, under the selected output/seed:

- benchmark_contract.json: exact train/validation subject IDs and
  normalization/hyperparameter settings; refuses mismatched resumes.
- normalization_summary.json: per-modality training-fitted scale diagnostics.
- leaderboard.csv: one row per model with subject-averaged and pooled
  balanced accuracy/F1, and the counts of predicted classes.
- each mode/validation_metrics.json: confusion matrices and metrics
  for s07, s14 and s21 independently.
- each mode/validation_predictions.csv: true/predicted trial labels,
  including subject and trial ID.
- each neural mode/best.pt: best checkpoint with training-only scaler.
- each neural mode/history.json: epoch losses/validation metrics.

The '--skip-existing' flag uses only results already present in the
chosen output directory with the same benchmark_contract.json.
Use a new '--out' directory for another hyperparameter configuration.
Do not select hyperparameters on the final test subject.

## Remaining methodological limitations

This experiment tests the **legacy extracted** pretrained features;
it does not establish correct EEGPT physical input conventions.
The trial aggregation averages the 4 within-window tokens before
fusion; alternative hierarchical/time-aware pooling can be evaluated
after a stable baseline. The validation subjects are selected from
a small cohort with nonidentical class prevalences. A good dev score
is not an unbiased population-generalization claim. If normalized
multimodal fusion fails, compare against signal-domain EEG/EDA/PPG
baselines before increasing architectural complexity.
