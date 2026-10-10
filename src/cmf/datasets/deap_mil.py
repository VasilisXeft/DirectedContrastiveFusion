"""Trial-level bags of cached DEAP windows. Never infer trial IDs from labels."""
from collections import defaultdict
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from cmf.datasets.common import CachedMultimodalDataset
from cmf.utils.io import load_npz


def _trial_key(row, meta):
    for key in ("trial_id", "trial", "video_id", "experiment_id"):
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key])
        if key in meta and meta[key] is not None and str(meta[key]).strip():
            return str(meta[key])
    # Only accept explicit trial tokens, never window index or class label.
    filename = str(row["file"])
    match = re.search(r"trial[_-]?([0-9]+)(?:[^0-9]|$)", filename, re.I)
    if match:
        return match.group(1)
    raise ValueError("Cannot identify trial for %s. Add trial_id/trial to manifest.csv or NPZ meta_json." % filename)


def _window_order(row, meta):
    for key in ("window_start", "start_sec", "start_time", "window_idx", "window_index", "start_sample"):
        value = row.get(key, None)
        if value is None or pd.isna(value):
            value = meta.get(key, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    # Sequential sample filenames are a fallback only; verify exporter order.
    match = re.search(r"sample_([0-9]+)[.]npz$", str(row["file"]), re.I)
    if match:
        return float(match.group(1))
    raise ValueError("Missing window ordering metadata for %s" % row["file"])


class DEAPTrialBagDataset(Dataset):
    def __init__(self, cache_dir, manifest, split, reliability_corruption=0.0, max_windows=None):
        self.windows = CachedMultimodalDataset(cache_dir, split=split, manifest=manifest,
                                                 reliability_corruption=reliability_corruption)
        groups = defaultdict(list)
        labels = {}
        for i, row in self.windows.manifest.iterrows():
            _, target, meta = load_npz(self.windows.cache_dir / row["file"])
            subject = str(row.get("subject", meta.get("subject", "")))
            if not subject:
                raise ValueError("Missing subject for %s" % row["file"])
            key = (subject, _trial_key(row, meta))
            label = int(np.asarray(target).reshape(-1)[0])
            if key in labels and labels[key] != label:
                raise ValueError("Inconsistent trial label for %s" % (key,))
            labels[key] = label
            groups[key].append((_window_order(row, meta), i))
        self.bags = []
        for key, entries in sorted(groups.items()):
            entries.sort(key=lambda x: x[0])
            starts = [x[0] for x in entries]
            if len(starts) != len(set(starts)):
                raise ValueError("Duplicate window timestamps for trial %s" % (key,))
            if max_windows is not None and len(entries) > max_windows:
                raise ValueError("Trial %s has %d windows, exceeding max_windows=%d" %
                                 (key, len(entries), max_windows))
            self.bags.append((key, [i for _, i in entries], labels[key]))
        if not self.bags:
            raise ValueError("No trial bags found")
        print("%s: %d trials, %d windows, bag lengths=%s" %
              (split, len(self.bags), len(self.windows),
               sorted(set(len(indices) for _, indices, _ in self.bags))))

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        key, indices, label = self.bags[index]
        samples = [self.windows[i] for i in indices]
        names = sorted(samples[0]["modalities"])
        return {
            "modalities": {n: torch.stack([s["modalities"][n] for s in samples]) for n in names},
            "present": {n: torch.tensor([s["present"][n] for s in samples]) for n in names},
            "quality_penalty": {n: torch.tensor([s["quality_penalty"][n] for s in samples]) for n in names},
            "target": torch.tensor(label, dtype=torch.long),
            "key": key,
        }


def collate_trial_bags(batch):
    names = sorted(batch[0]["modalities"])
    b = len(batch)
    max_len = max(len(x["modalities"][names[0]]) for x in batch)
    lengths = torch.tensor([len(x["modalities"][names[0]]) for x in batch])
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    mods, present, quality = {}, {}, {}
    for n in names:
        shape = batch[0]["modalities"][n].shape[1:]
        mods[n] = torch.zeros((b, max_len, *shape), dtype=batch[0]["modalities"][n].dtype)
        present[n] = torch.zeros(b, max_len)
        quality[n] = torch.ones(b, max_len)
        for i, item in enumerate(batch):
            t = lengths[i].item()
            mods[n][i, :t] = item["modalities"][n]
            present[n][i, :t] = item["present"][n]
            quality[n][i, :t] = item["quality_penalty"][n]
    return {"modalities": mods, "present": present, "quality_penalty": quality,
            "mask": mask, "target": torch.stack([x["target"] for x in batch]),
            "keys": [x["key"] for x in batch]}
