from pathlib import Path
import pickle
import numpy as np

from cmf.utils.io import save_sample


EEG_CHANNELS = slice(0, 32)
EDA_CHANNEL = 36
PPG_CHANNEL = 38
TEMP_CHANNEL = 39
FS = 128


def _windows(n, win, stride):
    for start in range(0, n - win + 1, stride):
        yield start, start + win


def preprocess_deap(cfg):
    """Prepare DEAP physiological windows in the common CMF cache format.

    Raw paths are supplied only through the YAML config; no dataset path is
    hard-coded in the repository. Face video is optional and is deliberately
    not decoded here: visual embeddings can be added in a separate extraction
    stage without duplicating the physiological cache.
    """
    dcfg = cfg["dataset"]
    data_dir = Path(dcfg["data_path"]).expanduser()
    video_dir = Path(dcfg["video_path"]).expanduser() if dcfg.get("video_path") else None
    out = Path(dcfg.get("cache_dir", "data/processed/deap"))
    out.mkdir(parents=True, exist_ok=True)

    window_s = float(dcfg.get("window_seconds", 10))
    stride_s = float(dcfg.get("stride_seconds", 5))
    baseline_s = float(dcfg.get("baseline_seconds", 3))
    label_name = str(dcfg.get("label", "valence")).lower()
    threshold = float(dcfg.get("binary_threshold", 5.0))

    label_idx = {"valence": 0, "arousal": 1}
    if label_name not in label_idx:
        raise ValueError("DEAP label must be 'valence' or 'arousal'.")

    win = int(round(window_s * FS))
    stride = int(round(stride_s * FS))
    baseline = int(round(baseline_s * FS))

    rows = []
    sample_id = 0
    dat_files = sorted(data_dir.glob("s*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No DEAP .dat files found in {data_dir}")

    for dat_path in dat_files:
        subject = dat_path.stem
        with dat_path.open("rb") as f:
            obj = pickle.load(f, encoding="latin1")
        data = np.asarray(obj["data"])
        labels = np.asarray(obj["labels"])

        if data.shape[0] != 40 or data.shape[1] < 40:
            raise ValueError(f"Unexpected DEAP shape for {dat_path}: {data.shape}")

        for trial in range(data.shape[0]):
            x = data[trial, :, baseline:].astype(np.float32, copy=False)
            y_score = float(labels[trial, label_idx[label_name]])
            y = np.asarray(int(y_score >= threshold), dtype=np.int64)

            video_path = None
            if video_dir is not None:
                candidate = video_dir / subject / f"{subject}_trial{trial+1:02d}.avi"
                if candidate.exists():
                    video_path = str(candidate)

            for start, end in _windows(x.shape[1], win, stride):
                mods = {
                    "eeg": x[EEG_CHANNELS, start:end].T,
                    "eda": x[EDA_CHANNEL, start:end, None],
                    "ppg": x[PPG_CHANNEL, start:end, None],
                    "temp": x[TEMP_CHANNEL, start:end, None],
                }
                meta = {
                    "subject": subject,
                    "trial": int(trial + 1),
                    "window_start_s": float(start / FS),
                    "window_end_s": float(end / FS),
                    "label_score": y_score,
                    "label_name": label_name,
                    "video_path": video_path,
                    "missing_modalities": [],
                }
                fname = f"sample_{sample_id:06d}.npz"
                save_sample(out / fname, mods, y, meta)
                rows.append((fname, subject))
                sample_id += 1

    # Subject-aware split placeholder. LOSO scripts should override this using
    # metadata/manifest rather than random window splits.
    import pandas as pd
    manifest = pd.DataFrame(rows, columns=["file", "subject"])
    manifest["split"] = "train"
    manifest.to_csv(out / "manifest.csv", index=False)
    return out
