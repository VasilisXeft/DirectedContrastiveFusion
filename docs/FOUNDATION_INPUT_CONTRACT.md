# Foundation-model input contract, validation and normalization

This separates **raw input preprocessing before an encoder** from **standardization of already-extracted features before fusion**. They are distinct operations.

## Official pretrained model contracts

| Backbone | Verified contract | Legacy DEAP extraction | Conclusion |
|---|---|---|---|
| EEGPT (braindecode/eegpt-pretrained) | Tensor shape [B,C,T], named EEG channels. Released HF checkpoint config: 62 channels, 250 Hz and 1000 samples; 512-D token embeddings. | 32 named DEAP EEG electrodes, 128 to 256 Hz, 1024 samples per 4-second crop, EEG data multiplied by 1e-6, output [4,512]. | Checkpoint *sampling metadata differs*. EEG amplitude units are not sufficiently verified. |
| MOMENT-1-large | Float tensor [B,C,512], optional [B,512] valid mask; embedding task initialized with model.init(); RevIN internally normalizes each input series. 1024-D pooled output. | DEAP PPG, EDA and TEMP standardized over 10 seconds, divided into four 2.5-second chunks, each resampled to 512 steps, output [4,1024]. | Valid input shape, but external z-score is redundant. The short chunk duration and internally removed absolute levels may limit EDA/TEMP utility. |
| DINOv2-small | RGB images, rescale 1/255, resize shortest side 256, center crop 224, ImageNet mean/std normalization. CLS output 384-D. | OpenCV BGR converted to RGB uint8, official Hugging Face AutoImageProcessor applied to four frames/window, output [4,384]. | Numerically correct image preprocessing, but no face detector: center crop can omit relevant facial content. |

Authoritative sources:
- [Released EEGPT checkpoint config](https://huggingface.co/braindecode/eegpt-pretrained/blob/main/config.json)
- [Braindecode EEGPT implementation](https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegpt.py)
- [Original EEGPT pretraining pipeline](https://github.com/BINE022/EEGPT/blob/main/datasets/pretrain/prepare_pretrain_dataset.py)
- [MOMENT-1-large checkpoint config](https://huggingface.co/AutonLab/MOMENT-1-large/blob/main/config.json)
- [MOMENT embedding logic](https://github.com/moment-timeseries-foundation-model/moment/blob/main/momentfm/models/moment.py)
- [MOMENT RevIN normalization](https://github.com/moment-timeseries-foundation-model/moment/blob/main/momentfm/models/layers/revin.py)
- [DINOv2 image processor](https://huggingface.co/facebook/dinov2-small/blob/main/preprocessor_config.json)

### EEGPT scaling remains a scientific concern

Stored DEAP EEG p95 amplitudes near 8, 51 and 21 (for the inspected subjects) look plausible in microvolts. The original EEGPT pretraining scripts convert different raw dataset units into a **millivolt-like** scale, for example microvolt numbers divided by 1000 or volt numbers multiplied by 1000. Our legacy adapter converts DEAP numbers with a factor of 1e-6; that is *1000-fold smaller* than dividing by 1000.

This is **not yet conclusive proof** of the exact amplitude convention expected by the Braindecode-hosted checkpoint. Do not silently regenerate the entire EEG cache without verifying units and conducting a controlled development-only input ablation. The hosted checkpoint declares 250 Hz/1000 samples whereas the original EEGPT README describes 256 Hz/1024 samples, approximately the same four-second duration.

Our DEAP extraction uses EEG channels 0-31 (Geneva electrode order), GSR/EDA channel 36, plethysmograph/PPG channel 38, and temperature channel 39. All channels in the distributed preprocessed arrays share the 128 Hz sampling rate. The first three seconds are discarded to align with the 60-second facial recording. Equal durations do **not** prove exact video synchronization.

## Run checkpoint-backed input audit

Pull the branch, then inspect a real local DEAP trial using:

    python scripts/audit_deap_foundation_inputs.py --deap-dir "C:/DEAP/data_preprocessed_python" --video-manifest data/deap_video_manifest.csv

The script reads the checkpoint configuration JSON files and DINOv2 processor configuration, prints channel coverage and input mismatches, inspects raw DEAP amplitudes, optional video timing/geometry and existing cache feature magnitudes. It neither fits on labels nor modifies the data.

## Fold-fitted embedding normalization: enabled

The foundation benchmark YAML now enables independent per-modality, per-feature **training-only** z-scores using minimum std 1e-5 and optional standardized-value clip 8. In every 18/3/1 LOSO fold:

1. Fit means and population standard deviations for EEG, PPG, EDA, TEMP and FACE using **only the 18 training subjects** and all training-window tokens.
2. Standardize each feature dimension consistently across tokens and across subjects using those *training* statistics.
3. Apply the same frozen scalers to the 3 validation subjects and 1 test subject. Never compute held-out normalization statistics.
4. Estimate feature-level quality penalties on the **original cached representation**, before applying normalization.
5. Save normalization tensors under the feature_normalization key of each best.pt checkpoint; reload those saved tensors at test time. Write a feature_normalization_summary.json for inspection.
6. Leave all cached NPZ arrays unmodified.

This fixes cross-modal *scale imbalance* as seen by fusion but does not restore information lost in backbone inference. A prior linear probe already standardizes features, so its cross-subject results still reflect a substantial representation/domain-shift challenge.

Run the normalization and LOSO split tests:

    python -m unittest discover -s tests -p test_feature_normalization.py
    python -m unittest discover -s tests -p test_loso_validation_split.py

## Optional safe new-extraction preprocessing ablation

The extraction script now accepts these explicit settings:
- EEG target sampling 250 (hosted checkpoint) vs legacy 256
- EEG multiplier 0.001 (a millivolt-scale candidate) vs legacy 0.000001
- MOMENT input-normalization none (let internal RevIN operate directly) vs legacy_window_zscore

Any different settings **must use a new cache output directory**. The extractor writes a preprocessing_contract.json and refuses to mix incompatible extraction settings into the same cache. Its default settings reproduce the old cache.

Example small development-only extraction with new settings:

    python scripts/extract_deap_foundation_5m.py --deap-dir "C:/DEAP/data_preprocessed_python" --video-manifest data/deap_video_manifest.csv --subjects s01 --allow-missing-videos --eeg-target-sfreq 250 --eeg-input-scale 0.001 --moment-input-normalization none --out data/features/deap_foundation_input_probe_s01

Do not re-extract the full cohort yet. Confirm the audit results, fit and test subject-independent normalization, then evaluate encoder-input ablations on development subjects before final benchmark runs.

Finally, DEAP's valence label is a **60-second trial rating**, not independently assessed for every 10-second window. Trial-level evaluation should be the primary next experiment; window-level labels constitute weak supervision.

## Unsupervised EEGPT input-scale sensitivity probe

Without recomputing MOMENT or DINOv2, run this *development-only*
probe after auditing the checkpoint and raw data:

    python scripts/probe_eegpt_input_scales.py --deap-dir "C:/DEAP/data_preprocessed_python" --subjects s01 s02 --target-sfreq 250

It reuses the same frozen EEGPT checkpoint to generate embeddings from the
same two subjects and windows at two candidate scales (1e-6 and 1e-3).
Reports output variance and relative embedding changes. This is a **sensitivity
test**, not a supervised comparison or proof that larger variance is better.
Neither validation nor final test labels are used, and no cache is changed.

## Empirical input-scale audit (s01/s02, 12 sampled 10-second windows)

The local adapter loaded all 102 expected EEGPT pretrained tensors; all
32 DEAP channels matched the EEGPT channel vocabulary. Both 250-Hz
and 256-Hz input configurations pass their deterministic shape/scale
unit tests. At **250 Hz on the same 12 windows**:

- Legacy input multiplier 1e-6: mean between-window embedding feature
  standard deviation 0.00081824; embedding mean L2 norm 26.517.
- Candidate input multiplier 1e-3: between-window feature standard
  deviation 0.1481348; mean embedding norm 22.228.
- Mean paired embedding L2 difference 18.8616, or 0.7113 times the
  mean legacy embedding norm.

The variability increases approximately **181-fold**. This establishes
that **raw EEG input scaling drastically changes what pretrained EEGPT
encodes**; it does *not* establish better emotion recognition, because
neither labels nor subject-disjoint performance were assessed in this
small unlabeled pilot.

### Paired, trial-level, development-only input ablation

New script: [compare_deap_eegpt_inputs.py](../scripts/compare_deap_eegpt_inputs.py)

It compares THREE EEGPT candidates on the same DEAP trials and 10-second
windows, holding backbone weights fixed:

- `cached_256_1e-6`: uses old frozen cache at 256 Hz / 1e-6.
- `eegpt_250_1e-6`: recomputes only EEGPT at 250 Hz / 1e-6.
- `eegpt_250_1e-3`: recomputes only EEGPT at 250 Hz / 1e-3.

Both new arms share a model instance, raw trial data, window starts,
and pooling. The default is three 10-second windows per trial (starts
0, 25, 50 s); the existing cached baseline is selected at those *same*
starts. You may use `--windows-per-trial 6` or `11`, at greater compute
cost. Output is one averaged 512-D vector per trial, matching the
one-rating-per-60s DEAP task.

Evaluation is a paired 21-fold *development-only* subject LOSO with
a **fixed** RidgeClassifier(alpha=100, class_weight=balanced). On each
fold, StandardScaler is fit on training trials only. The final `s22`
subject is never opened. Each subject's new embeddings are saved in an
independent resume-safe cache under `runs/deap_foundation_5m/eegpt_input_comparison`;
neither existing five-modality NPZs nor their manifest are changed.

Commands:

    python -m unittest discover -s tests -p test_compare_deap_eegpt_inputs.py
    python scripts/compare_deap_eegpt_inputs.py --deap-dir "C:/DEAP/data_preprocessed_python" --dry-run
    python scripts/compare_deap_eegpt_inputs.py --deap-dir "C:/DEAP/data_preprocessed_python"

Compare the **paired per-subject** differences, mean and standard
deviation of balanced accuracy / macro-F1. More variable EEGPT
embeddings alone are insufficient evidence to change preprocessing.
If the 250 Hz / 1e-3 arm clearly transfers better, re-extract **EEGPT
only**, then build a separate five-modality cache while keeping the
existing MOMENT/DINOv2 arrays and preserving a documented
preprocessing manifest.

Caution: this script freezes the sampled window count and selected
subjects in the subject-level cache contract; choosing a different
window count or model requires a different `--out` directory.
