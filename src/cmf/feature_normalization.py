"""Fold-fitted normalization of frozen multimodal token embeddings.

Each modality has independent per-feature training mean/std. The same statistics
are applied to every token from that modality in train/validation/test.
No labels or held-out-subject features are consulted during fitting.
"""
from pathlib import Path

import numpy as np
import torch


class FeatureStandardizer:
    def __init__(self, min_std=1e-5, clip=8.0):
        self.min_std = float(min_std)
        self.clip = float(clip) if clip is not None else None
        if self.min_std <= 0 or not np.isfinite(self.min_std):
            raise ValueError("min_std must be finite and positive")
        if self.clip is not None and (self.clip <= 0 or not np.isfinite(self.clip)):
            raise ValueError("clip must be None or finite and positive")
        self.stats = {}

    def fit(self, cache_dir, train_manifest, modalities=None):
        """Fit using only rows explicitly supplied from the training split.

        Uses numerically stable batch-Welford aggregation over all tokens and
        windows, without storing or modifying the complete cache.
        """
        cache_dir = Path(cache_dir)
        if len(train_manifest) == 0:
            raise ValueError("Cannot fit feature normalization on an empty split")
        if "split" in train_manifest.columns:
            splits = set(train_manifest["split"].astype(str))
            if splits != {"train"}:
                raise ValueError(
                    f"Feature normalization fit requires train-only rows, got {sorted(splits)}"
                )
        required = set(modalities) if modalities is not None else None
        running = {}
        for filename in train_manifest["file"]:
            with np.load(cache_dir / str(filename), allow_pickle=False) as npz:
                available = {key[3:] for key in npz.files if key.startswith("x__")}
                names = required if required is not None else available
                if required is not None and not names.issubset(available):
                    raise ValueError(f"Missing modalities in {filename}: {names - available}")
                for modality in sorted(names):
                    array = np.asarray(npz[f"x__{modality}"], dtype=np.float64)
                    if array.ndim < 1:
                        raise ValueError(f"Invalid feature rank for {modality} in {filename}")
                    vectors = array.reshape(-1, array.shape[-1])
                    if vectors.shape[0] == 0 or not np.isfinite(vectors).all():
                        raise ValueError(f"Empty or non-finite features in {filename}:{modality}")
                    size = vectors.shape[0]
                    batch_mean = vectors.mean(axis=0)
                    batch_m2 = np.square(vectors - batch_mean).sum(axis=0)
                    if modality not in running:
                        running[modality] = [size, batch_mean, batch_m2]
                    else:
                        n, mean, m2 = running[modality]
                        if mean.shape != batch_mean.shape:
                            raise ValueError(f"Inconsistent {modality} embedding dimension")
                        combined = n + size
                        delta = batch_mean - mean
                        mean += delta * (size / combined)
                        m2 += batch_m2 + np.square(delta) * (n * size / combined)
                        running[modality][0] = combined
        if required is not None and set(running) != required:
            raise ValueError(f"Missing modalities after fitting: {required - set(running)}")
        self.stats = {}
        for modality, (n, mean, m2) in running.items():
            std = np.sqrt(np.maximum(m2 / n, 0.0))
            scale = np.maximum(std, self.min_std)
            self.stats[modality] = {
                "mean": mean.astype(np.float32),
                "scale": scale.astype(np.float32),
                "count": int(n),
            }
        return self

    def transform(self, modality, features):
        if modality not in self.stats:
            raise KeyError(f"No training-fitted normalization for {modality}")
        arr = np.asarray(features, dtype=np.float32)
        stats = self.stats[modality]
        if arr.shape[-1] != stats["mean"].shape[0]:
            raise ValueError(f"Bad {modality} feature dimension: {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite features in {modality} before normalization")
        scaled = (arr - stats["mean"]) / stats["scale"]
        if self.clip is not None:
            scaled = np.clip(scaled, -self.clip, self.clip)
        if not np.isfinite(scaled).all():
            raise ValueError(f"Non-finite features in {modality} after normalization")
        return scaled.astype(np.float32, copy=False)

    def state_dict(self):
        """Only tensors and primitive values; safe inside PyTorch checkpoint."""
        return {
            "method": "train_subject_per_feature_zscore",
            "min_std": self.min_std,
            "clip": self.clip,
            "stats": {
                mod: {
                    "mean": torch.from_numpy(item["mean"].copy()),
                    "scale": torch.from_numpy(item["scale"].copy()),
                    "count": item["count"],
                }
                for mod, item in self.stats.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("method") != "train_subject_per_feature_zscore":
            raise ValueError("Unknown feature-normalization checkpoint format")
        obj = cls(min_std=state["min_std"], clip=state["clip"])
        obj.stats = {
            mod: {
                "mean": torch.as_tensor(item["mean"]).cpu().numpy().copy(),
                "scale": torch.as_tensor(item["scale"]).cpu().numpy().copy(),
                "count": int(item["count"]),
            }
            for mod, item in state["stats"].items()
        }
        return obj

    def summary(self):
        return {
            modality: {
                "tokens": item["count"],
                "feature_dim": int(item["mean"].size),
                "std_median": float(np.median(item["scale"])),
                "std_min": float(np.min(item["scale"])),
                "std_max": float(np.max(item["scale"])),
                "floor_fraction": float(np.mean(item["scale"] <= self.min_std)),
            }
            for modality, item in self.stats.items()
        }
