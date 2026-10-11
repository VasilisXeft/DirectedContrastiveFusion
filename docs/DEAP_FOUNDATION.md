# DEAP foundation-model benchmark (experimental)

## Requirements
Install the repository and optional inference dependencies:

```powershell
pip install -e .
pip install momentfm transformers pillow
```

Use a compatible version of PyTorch/Transformers for the installed MOMENT release.

## Data prerequisites
- Standard DEAP preprocessed Python files `s01.dat` ... `s32.dat`.
- Access to DEAP face videos **only for subjects/trials that actually have videos**. The standard DEAP face-video subset does not cover all 32 participants. **A complete 32-subject five-modality LOSO cannot be claimed from the original dataset without missing-video handling.**
- A CSV mapping each video to its exact subject/trial. Columns: `subject,trial,video_path`, where `trial` is zero-based. Video paths should be absolute, or relative to the current working directory.
- A verified, pretrained EEGPT TorchScript **feature extractor**, taking [B,32,T] and returning [B,D] or [B,tokens,D]. A classifier head or randomly initialized model is not a substitute. Confirm its expected sampling rate, channel montage and preprocessing; adapt raw EEG accordingly before running extraction. The supplied script currently assumes 128-Hz DEAP-preprocessed EEG and does not automatically implement EEGPT's checkpoint-specific channel remapping.

## Extraction
```powershell
python scripts/extract_deap_foundation_5m.py --deap-dir "D:/DEAP/data_preprocessed_python" --video-manifest "D:/DEAP/video_manifest.csv" --eegpt-torchscript "D:/models/eegpt_features.ts" --subjects s01
```

**Do not run extraction until checking the exact video timing.** DEAP physiological trials include 3 s baseline, which is removed by this extractor. If videos start before/after that baseline, adjust the video timestamp offset in the script. The script currently assumes the video begins at the same time as the post-baseline signal. Duration checking alone cannot verify synchronization.

The script saves `x__eeg`, `x__ppg`, `x__eda`, `x__temp`, `x__face`, scalar labels, and metadata in compressed NPZ files and a manifest CSV.

## Shape contract
Expected token dimensions in `configs/deap_foundation_5m.yaml`:
- EEGPT: 768 (verify against your checkpoint)
- MOMENT large: 1024 (verify against your checkpoint)
- DINOv2 small: 384

Inspect the first NPZ and adjust config if a checkpoint produces a different feature width.

## Quick subject-wise test
```powershell
python scripts/run_deap_external_5m.py --config configs/deap_foundation_5m.yaml --mode unimodal_eeg --subject s01 --seed 42
python scripts/run_deap_external_5m.py --config configs/deap_foundation_5m.yaml --mode late --subject s01 --seed 42
python scripts/run_deap_external_5m.py --config configs/deap_foundation_5m.yaml --mode full --subject s01 --seed 42
```

Full LOSO is obtained by omitting `--subject`, **but only after** the cache covers the intended evaluation cohort and the video alignment is verified. For a proper full 32-subject experiment, use a missing-modality design or an EEG/physiology-only cohort; the present extractor intentionally fails on missing video rather than silently inventing face features.

## Validation rules
- Never use test-subject labels to select checkpoints.
- Cache extraction is frozen/pretrained-only, with no fitting to held-out subjects.
- For publications, report the actual subject subset, per-subject results, and the original data/video alignment.
- Check that the selected pretrained weights were not previously fine-tuned on DEAP labels; otherwise report potential pretraining overlap.
