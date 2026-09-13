from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

from cmf.utils.io import load_npz


def resample_numeric(x, length):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] == length:
        return x
    if x.shape[0] < 2:
        return np.repeat(x[:1], length, axis=0)
    old = np.linspace(0, 1, x.shape[0])
    new = np.linspace(0, 1, length)
    f = interp1d(old, x, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(new).astype(np.float32)


def sample_video_frames(path, n_frames=16, image_size=112):
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frames from {path}")
    idxs = np.linspace(0, total - 1, n_frames).astype(int)
    frames, wanted = [], set(idxs.tolist())
    i = 0
    while cap.isOpened() and len(frames) < len(idxs):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames[:n_frames], axis=0).astype(np.uint8)


def sample_image_volume(x, n_frames=16, image_size=112):
    import cv2
    x = np.asarray(x)
    # Guess temporal axis as the smallest/last plausible dimension.
    if x.ndim == 3:
        t_axis = 2 if x.shape[2] <= max(x.shape[0], x.shape[1]) else 0
        x = np.moveaxis(x, t_axis, 0)
        x = x[..., None]
    elif x.ndim == 4 and x.shape[-1] not in (1, 3):
        x = np.moveaxis(x, -1, 0)
    idxs = np.linspace(0, x.shape[0]-1, n_frames).astype(int)
    out = []
    for i in idxs:
        fr = x[i]
        if fr.ndim == 2:
            fr = fr[..., None]
        if fr.shape[-1] == 1:
            fr = np.repeat(fr, 3, axis=-1)
        fr = fr.astype(np.float32)
        lo, hi = np.nanpercentile(fr, [1, 99])
        fr = np.clip((fr - lo) / (hi - lo + 1e-6), 0, 1)
        fr = (fr * 255).astype(np.uint8)
        fr = cv2.resize(fr, (image_size, image_size), interpolation=cv2.INTER_AREA)
        out.append(fr)
    return np.stack(out, axis=0)


class CachedMultimodalDataset(Dataset):
    def __init__(self, cache_dir, split="train", modality_dropout=0.0):
        self.cache_dir = Path(cache_dir)
        self.manifest = pd.read_csv(self.cache_dir / "manifest.csv")
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.modality_dropout = float(modality_dropout)
        if len(self.manifest) == 0:
            raise ValueError(f"No samples for split={split} in {self.cache_dir / 'manifest.csv'}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        mods, target, meta = load_npz(self.cache_dir / row["file"])
        out = {}
        present = {}
        missing = set(meta.get("missing_modalities", []))
        for name, x in mods.items():
            naturally_missing = name in missing
            drop = (not naturally_missing) and self.modality_dropout > 0 and np.random.rand() < self.modality_dropout
            present[name] = 0.0 if (drop or naturally_missing) else 1.0
            if x.ndim == 4:  # [T,H,W,C]
                t = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0
            else:
                t = torch.from_numpy(x).float()
            if drop:
                t = torch.zeros_like(t)
            out[name] = t
        y = torch.as_tensor(target)
        return {"modalities": out, "present": present, "target": y, "meta": meta}


def collate_multimodal(batch):
    names = sorted(batch[0]["modalities"].keys())
    modalities = {n: torch.stack([b["modalities"][n] for b in batch]) for n in names}
    present = {n: torch.tensor([b["present"][n] for b in batch], dtype=torch.float32) for n in names}
    targets = torch.stack([b["target"] for b in batch])
    return {"modalities": modalities, "present": present, "target": targets, "meta": [b["meta"] for b in batch]}
