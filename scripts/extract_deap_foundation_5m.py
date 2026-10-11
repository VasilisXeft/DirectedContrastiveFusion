"""Extract frozen foundation-model embeddings from raw DEAP Python trials.

Requires: pip install momentfm transformers pillow braindecode
EEGPT checkpoints vary by release: provide a TorchScript EEGPT feature extractor
that accepts [B, C, T] and returns [B, D] or [B, tokens, D].
No randomly initialized fallback is permitted.
"""
import argparse
import csv
import pickle
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
from cmf.utils.io import ensure_dir, save_npz

EEG_CHANNELS = 32
# DEAP preprocessed channel indices (40 channels, 128 Hz).
PPG_INDEX = 38  # plethysmograph
EDA_INDEX = 36  # GSR
TEMP_INDEX = 39 # temperature


def windows(n, size, stride):
    return [(i, i + size) for i in range(0, n - size + 1, stride)]


def standardize(x):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num((x - x.mean(axis=-1, keepdims=True)) /
                         (x.std(axis=-1, keepdims=True) + 1e-6))


class EEGPTFeatures:
    def __init__(self, path, device, output_tokens=4):
        self.device = device
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.output_tokens = output_tokens

    @torch.inference_mode()
    def __call__(self, signal):
        x = torch.from_numpy(standardize(signal)).unsqueeze(0).to(self.device)
        y = self.model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if isinstance(y, dict):
            y = y.get("last_hidden_state", y.get("features"))
        if y is None:
            raise ValueError("EEGPT wrapper did not return embeddings")
        if y.ndim == 2:
            y = y.unsqueeze(1)
        if y.ndim != 3:
            raise ValueError(f"Unexpected EEGPT output shape {tuple(y.shape)}")
        y = torch.nn.functional.adaptive_avg_pool1d(y.transpose(1, 2), self.output_tokens).transpose(1, 2)
        return y[0].float().cpu().numpy()


class MomentFeatures:
    def __init__(self, device, model_id="AutonLab/MOMENT-1-large", length=512, tokens=4):
        from momentfm import MOMENTPipeline
        self.model = MOMENTPipeline.from_pretrained(
            model_id, model_kwargs={"task_name": "embedding"}).to(device).eval()
        self.device, self.length, self.tokens = device, length, tokens

    @torch.inference_mode()
    def __call__(self, signal):
        from scipy.signal import resample
        x = standardize(signal.reshape(1, -1))
        # Pool distinct temporal chunks to preserve temporal ordering.
        chunks = np.array_split(x, self.tokens, axis=-1)
        embeddings = []
        for chunk in chunks:
            seq = resample(chunk, self.length, axis=-1).astype(np.float32)
            tensor = torch.from_numpy(seq[None]).to(self.device)
            output = self.model(x_enc=tensor)
            emb = output.embeddings
            embeddings.append(emb.reshape(-1).float().cpu().numpy())
        return np.stack(embeddings)


class DinoFeatures:
    def __init__(self, device, model_id="facebook/dinov2-small", tokens=4):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device, self.tokens = device, tokens

    @torch.inference_mode()
    def __call__(self, frames):
        from PIL import Image
        picks = np.linspace(0, len(frames) - 1, self.tokens).round().astype(int)
        images = [Image.fromarray(frames[i].astype(np.uint8)).convert("RGB") for i in picks]
        batch = self.processor(images=images, return_tensors="pt").to(self.device)
        result = self.model(**batch).last_hidden_state[:, 0]
        return result.float().cpu().numpy()


def load_video(path, n_frames=24):
    import cv2
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        raise ValueError(f"Cannot decode video {path}")
    frames = []
    for idx in np.linspace(0, count - 1, n_frames).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Missing frame {idx} in {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deap-dir", required=True, help="Folder with s01.dat ... s32.dat")
    p.add_argument("--video-manifest", required=True, help="CSV: subject,trial,video_path")
    p.add_argument("--eegpt-torchscript", required=True, help="Pretrained EEGPT TorchScript feature wrapper")
    p.add_argument("--out", default="data/features/deap_foundation_5m")
    p.add_argument("--subjects", nargs="+", default=None)\n    p.add_argument("--allow-missing-videos", action="store_true", help="Skip trials without video")
    p.add_argument("--window-seconds", type=int, default=10)
    p.add_argument("--stride-seconds", type=int, default=5)
    p.add_argument("--threshold", type=float, default=5.)\n    p.add_argument("--moment-model", default="AutonLab/MOMENT-1-large")\n    p.add_argument("--dinov2-model", default="facebook/dinov2-small")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    root = Path(a.deap_dir)
    output = ensure_dir(a.out)
    with open(a.video_manifest, newline="", encoding="utf-8-sig") as f:
        mapping = {(row["subject"].lower().replace("s", "").zfill(2),
                    int(row["trial"])): Path(row["video_path"]) for row in csv.DictReader(f)}
    eeg = EEGPTFeatures(a.eegpt_torchscript, a.device)
    moment = MomentFeatures(a.device)
    dino = DinoFeatures(a.device)
    subjects = a.subjects or sorted({f"s{int(s):02d}" for s, _ in mapping})
    manifest_path = output / "manifest.csv"
    existing = {}
    if manifest_path.exists():
        with manifest_path.open(newline="") as f:
            existing = {r["file"]: r for r in csv.DictReader(f)}
    for subject in subjects:
        subject = f"s{int(subject.lower().replace('s', '')):02d}"
        with (root / f"{subject}.dat").open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        signals, labels = data["data"], data["labels"]
        for trial in range(len(signals)):
            video = mapping.get((subject[1:], trial))
            if video is None or not video.is_file():
                raise FileNotFoundError(f"Video mapping missing: {subject}, trial={trial} (zero-based)")
            # IMPORTANT: manifest videos must be pre-aligned to the full trial.
            # DEAP physiological data contain 3s baseline; discard it.
            raw = signals[trial, :, 3 * 128:]
            n = raw.shape[-1]
            # Videos are sampled uniformly over the trial; crop to the corresponding window.
            cap = __import__("cv2").VideoCapture(str(video))
            duration = cap.get(__import__("cv2").CAP_PROP_FRAME_COUNT) / max(cap.get(__import__("cv2").CAP_PROP_FPS), 1)
            cap.release()
            if abs(duration - n / 128) > 1:
                raise ValueError(f"Video duration mismatch for {subject} trial {trial}: {duration:.1f}s vs {n/128:.1f}s")
            for start, end in windows(n, a.window_seconds * 128, a.stride_seconds * 128):
                filename = f"{subject}_trial{trial:02d}_start{start:06d}.npz"
                if filename in existing and (output / filename).is_file():
                    continue
                # Windowed video extraction, not full-trial frames.
                import cv2
                cap = cv2.VideoCapture(str(video))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = []
                for t in np.linspace(start / 128, end / 128, 4, endpoint=False):
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000))
                    ok, fr = cap.read()
                    if not ok:
                        cap.release()
                        raise ValueError(f"Cannot read aligned video at {t:.2f}s: {video}")
                    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                cap.release()
                modalities = {
                    "eeg": eeg(raw[:EEG_CHANNELS, start:end]),
                    "ppg": moment(raw[PPG_INDEX, start:end]),
                    "eda": moment(raw[EDA_INDEX, start:end]),
                    "temp": moment(raw[TEMP_INDEX, start:end]),
                    "face": dino(frames),
                }
                if any(not np.isfinite(x).all() for x in modalities.values()):
                    raise ValueError(f"Nonfinite embedding {filename}")
                label = int(labels[trial, 0] > a.threshold)
                save_npz(output / filename, modalities, np.asarray(label, dtype=np.int64),
                         {"subject": subject, "trial": trial, "start": start, "end": end})
                existing[filename] = {"file": filename, "subject": subject, "trial": trial,
                                      "label": label, "split": "train"}
                # Persist incrementally to allow restart after interruption.
                with manifest_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["file", "subject", "trial", "label", "split"])
                    writer.writeheader()
                    writer.writerows(existing.values())
                print("EXTRACTED", filename, flush=True)


if __name__ == "__main__":
    main()
