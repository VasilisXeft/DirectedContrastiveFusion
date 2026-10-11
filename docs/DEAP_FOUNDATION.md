# DEAP foundation-model benchmark (experimental)

## Requirements
Install the repository and optional inference dependencies:

```powershell
pip install -e .
pip install momentfm transformers pillow braindecode huggingface_hub
```

Use a compatible version of PyTorch/Transformers for the installed MOMENT release.

## Data prerequisites
- Standard DEAP preprocessed Python files `s01.dat` ... `s32.dat`.
- Access to DEAP face videos **only for subjects/trials that actually have videos**. The standard DEAP face-video subset does not cover all 32 participants. **A complete 32-subject five-modality LOSO cannot be claimed from the original dataset without missing-video handling.**
- A CSV mapping each video to its exact subject/trial. Columns: `subject,trial,video_path`, where `trial` is zero-based. Video paths should be absolute, or relative to the current working directory.
- EEGPT pretrained weights downloaded from `braindecode/eegpt-pretrained`; the adapter maps DEAP channel names, resamples EEG to 256 Hz and extracts four 512-dimensional representations per window. The checkpoint channel-ID buffer is rebuilt for DEAP. Input amplitude units still require validation.

## Extraction
```powershell
python scripts/extract_deap_foundation_5m.py --deap-dir "D:/DEAP/data_preprocessed_python" --video-manifest "D:/DEAP/video_manifest.csv" --subjects s01
```

**Do not run extraction until checking the exact video timing.** DEAP physiological trials include 3 s baseline, which is removed by this extractor. If videos start before/after that baseline, adjust the video timestamp offset in the script. The script currently assumes the video begins at the same time as the post-baseline signal. Duration checking alone cannot verify synchronization.

The script saves `x__eeg`, `x__ppg`, `x__eda`, `x__temp`, `x__face`, scalar labels, and metadata in compressed NPZ files and a manifest CSV.

## Shape contract
Expected token dimensions in `configs/deap_foundation_5m.yaml`:
- EEGPT: 512 (verify against your checkpoint)
- MOMENT large: 1024 (verify against your checkpoint)
- DINOv2 small: 384

Inspect the first NPZ and adjust config if a checkpoint produces a different feature width.

## Quick subject-wise test
```powershell
python scripts/run_deap_external_5m.py --config configs/deap_foundation_5m.yaml --mode unimodal_eeg --subject s01 --seed 42
python scripts/run_deap_external_5m.py --config configs/deap_foundation_5m.yaml --mode late --subject s01 --seed 42
python scripts/run_deap_external_5m.py --config configs/deap_foundation_5m.yaml --mode full --subject s01 --seed 42
```

Full LOSO is obtained by omitting `--subject`, **but only after** the cache covers the intended evaluation cohort and the video alignment is verified. For a proper full 32-subject experiment, use a missing-modality design or an EEG/physiology-only cohort; the present extractor can skip missing videos with `--allow-missing-videos`, without inventing face features.

## Validation rules
- Never use test-subject labels to select checkpoints.
- Cache extraction is frozen/pretrained-only, with no fitting to held-out subjects.
- For publications, report the actual subject subset, per-subject results, and the original data/video alignment.
- Check that the selected pretrained weights were not previously fine-tuned on DEAP labels; otherwise report potential pretraining overlap.

## Direct pretrained EEGPT integration
The extractor uses native Braindecode EEGPT and the `braindecode/eegpt-pretrained` safetensors, excluding the pretrained `chans_id` buffer so that DEAP channel IDs can be reconstructed. It resamples 128-Hz signals to 256 Hz and extracts 4-second crop embeddings; no TorchScript export is needed. The existing cache was produced using DEAP EEG values multiplied by `1e-6`; this scaling must be verified against the checkpoint pretraining convention before interpreting results.

## 18/3/1 subject-independent validation (October 2026)
The foundation config uses `validation_strategy: spaced_previous` and
`validation_subject_count: 3`. For each outer LOSO test subject it selects
three distinct validation subjects by fixed circular offsets across the
subject list (1, 8, 15 for a 22-person cohort), leaving 18 training subjects.
This choice uses **subject IDs only**, not emotion labels or test performance.
Each subject participates as validation in three of the 22 LOSO folds.

For example, with `--subject s22`:
- Training: 18 participants, 7,865 windows (715 trials).
- Validation: `s21`, `s14`, `s07`, 1,309 windows (119 trials).
- Testing: `s22`, 440 windows (40 trials).

Inspect assignments without training:

```powershell
python scripts/run_deap_external_5m.py --config configs/deap_foundation_5m.yaml --subject s22 --dry-run
```

This changes the checkpoint path (e.g. `test_s22_val_s21-s14-s07`) so that
older single-validation checkpoints are not mistaken for new runs. The
selection criterion remains pooled **window-level macro-F1**. Because DEAP
ratings refer to 60-second trials, evaluating and selecting by trial-level
predictions is a proposed next methodological improvement. Three validation
subjects reduce dependence on one participant but cannot fix weak pretrained
representations or incompatible EEG amplitude preprocessing.

To keep `s22` as an untouched final evaluation subject during development,
do not inspect its metrics while tuning. A complete outer 22-fold LOSO
can be reported later with nested subject-disjoint validation, but selecting
hyperparameters after viewing outer test fold results introduces bias.
