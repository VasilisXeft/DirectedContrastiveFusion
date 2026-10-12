"""Trial-level DEAP dataset from existing five-modality frozen window embeddings.

One sample = one 60-second DEAP stimulus (11 overlapping 10-second windows).
Each cached window has four tokens per modality; the default pools those
four tokens into one vector, then stacks the 11 vectors chronologically.
Thus each model receives [11, feature_dim] per modality and ONE trial label.

No data are inferred from filenames alone for target verification: the cached
NPZ target must agree with the manifest. No encoder inference is performed.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


MODALITY_ORDER = ("eeg", "ppg", "eda", "temp", "face")
EXPECTED_WINDOW_STARTS = tuple(range(0, 6401, 640))


def parse_start(filename: str) -> int:
    match = re.search(r"_start(\d+)\.npz$", str(filename))
    if match is None:
        raise ValueError(f"Cached DEAP filename has no start offset: {filename}")
    return int(match.group(1))


def group_trials(window_manifest: pd.DataFrame, expected_starts=EXPECTED_WINDOW_STARTS):
    required = {"subject", "trial", "file", "label"}
    if not required.issubset(window_manifest.columns):
        raise ValueError(f"DEAP manifest missing columns: {sorted(required - set(window_manifest))}")
    if window_manifest["file"].duplicated().any():
        raise ValueError("Duplicate cache filenames detected")
    if window_manifest.empty:
        raise ValueError("No cached DEAP windows in the requested split")
    groups = []
    for (subject, trial), frame in window_manifest.groupby(["subject", "trial"], sort=True):
        labels = frame["label"].unique()
        if len(labels) != 1 or int(labels[0]) not in (0, 1):
            raise ValueError(f"Missing/conflicting binary labels for {subject} trial {trial}")
        positions = [parse_start(name) for name in frame["file"]]
        if sorted(positions) != sorted(expected_starts):
            raise ValueError(
                f"Incomplete or duplicated 60s trial {subject}/{trial}: "
                f"expected starts={list(expected_starts)}, got={sorted(positions)}"
            )
        ordered = frame.assign(_start=positions).sort_values("_start")
        groups.append({
            "subject": str(subject), "trial": int(trial), "label": int(labels[0]),
            "files": ordered["file"].astype(str).tolist(),
        })
    return groups


class DEAPTrialDataset(Dataset):
    """In-memory trial tensors, fitted train-only normalizer shared across splits."""

    def __init__(self, cache_dir, window_manifest, feature_normalizer=None,
                 modalities=MODALITY_ORDER, expected_starts=EXPECTED_WINDOW_STARTS):
        self.cache_dir = Path(cache_dir)
        self.modalities = tuple(modalities)
        if not self.modalities:
            raise ValueError("Provide at least one modality")
        self.groups = group_trials(window_manifest, expected_starts=expected_starts)
        self.records = []
        for group in self.groups:
            token_lists = {name: [] for name in self.modalities}
            for filename in group["files"]:
                with np.load(self.cache_dir / filename, allow_pickle=False) as z:
                    if int(np.asarray(z["target"]).item()) != group["label"]:
                        raise ValueError(f"Target mismatch: {filename}")
                    for name in self.modalities:
                        key = "x__" + name
                        if key not in z:
                            raise ValueError(f"{filename} missing modality {name}")
                        x = np.asarray(z[key], dtype=np.float32)
                        if x.ndim != 2 or x.shape[0] != 4 or not np.isfinite(x).all():
                            raise ValueError(f"{filename} unexpected {name} token array: {x.shape}")
                        # Transform EACH frozen token using training-only statistics
                        # before reducing it. This preserves semantics of clipping.
                        if feature_normalizer is not None:
                            x = feature_normalizer.transform(name, x)
                        token_lists[name].append(x.mean(axis=0).copy())
            tensors = {}
            for name, vectors in token_lists.items():
                x = np.stack(vectors).astype(np.float32)
                if x.shape[0] != len(expected_starts):
                    raise AssertionError("Wrong trial token count")
                tensors[name] = torch.from_numpy(x)
            self.records.append((group, tensors))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        group, tokens = self.records[index]
        return {
            "modalities": tokens,
            "present": {name: 1.0 for name in self.modalities},
            # Initial representation-only comparison, no learned quality gate.
            "quality_penalty": {name: 0.0 for name in self.modalities},
            "reliability_target": {name: 1.0 for name in self.modalities},
            "target": torch.tensor(group["label"], dtype=torch.long),
            "meta": {"subject": group["subject"], "trial": group["trial"],
                     "num_windows": len(group["files"])},
        }

    def shapes(self):
        return {
            name: tuple(self.records[0][1][name].shape)
            for name in self.modalities
        }

    def labels(self):
        return np.asarray([group["label"] for group, _ in self.records], dtype=np.int64)

    def subjects(self):
        return [group["subject"] for group, _ in self.records]
