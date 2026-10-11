"""Audit DEAP cached foundation embeddings and run leakage-safe trial-level probes.

Run before changing the fusion architecture:
    python scripts/diagnose_deap_foundation.py
This script reads cached embeddings; it does not modify training or use the test subject
for model selection. Only training subjects are used to fit the standardizer/probes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODALITIES = ("eeg", "ppg", "eda", "temp", "face")


def metric_line(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return (
        f"acc={accuracy_score(y, pred):.3f}, "
        f"bacc={balanced_accuracy_score(y, pred):.3f}, "
        f"macro_f1={f1_score(y, pred, average='macro', zero_division=0):.3f}, "
        f"true={np.bincount(y, minlength=2).tolist()}, "
        f"pred={np.bincount(pred, minlength=2).tolist()}, "
        f"cm={cm.tolist()}"
    )


def read_trial(root, rows):
    labels = {int(x) for x in rows["label"]}
    if len(labels) != 1:
        raise ValueError(f"Conflicting manifest labels within trial: {rows[['subject', 'trial']].iloc[0].to_dict()}")
    label = labels.pop()
    window_means = {name: [] for name in MODALITIES}
    for name in rows["file"]:
        with np.load(root / name, allow_pickle=False) as z:
            target = int(np.asarray(z["target"]).item())
            if target != label:
                raise ValueError(f"Manifest/NPZ label mismatch: {name}: {label} vs {target}")
            for modality in MODALITIES:
                key = "x__" + modality
                if key not in z:
                    raise ValueError(f"Missing modality {key}: {name}")
                x = z[key].astype(np.float32)
                if x.ndim != 2 or x.shape[0] != 4 or not np.isfinite(x).all():
                    raise ValueError(f"Invalid embeddings {name} {key}: {x.shape}")
                window_means[modality].append(x.mean(axis=0))
    pooled = {modality: np.stack(windows).mean(axis=0) for modality, windows in window_means.items()}
    window_spread = {modality: float(np.std(np.stack(windows), axis=0).mean())
                     for modality, windows in window_means.items()}
    return label, pooled, window_spread


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/features/deap_foundation_5m")
    p.add_argument("--val-subject", default="s21")
    p.add_argument("--test-subject", default="s22")
    p.add_argument("--alpha", type=float, default=100.)
    a = p.parse_args()

    root = Path(a.cache)
    df = pd.read_csv(root / "manifest.csv", dtype={"subject": str})
    expected_cols = {"subject", "trial", "file", "label"}
    if not expected_cols.issubset(df):
        raise ValueError(f"Missing manifest columns: {sorted(expected_cols - set(df))}")
    if df["file"].duplicated().any():
        raise ValueError("Duplicate cache file paths")
    missing = [str(root / x) for x in df["file"] if not (root / x).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing NPZ files; first: {missing[:3]}")
    if set(df["label"].unique()) - {0, 1}:
        raise ValueError("Expected binary labels 0 and 1")

    per_subject = df.groupby("subject", sort=True).agg(
        windows=("file", "size"), trials=("trial", "nunique"), positive_rate=("label", "mean")
    )
    print("\nSUBJECT WINDOW COUNTS AND CLASS BALANCE")
    print(per_subject.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"Total: {len(df)} windows, {len(per_subject)} subjects, "
          f"{df.groupby(['subject','trial']).ngroups} labeled trials")

    trial_records = []
    trial_features = {mod: [] for mod in MODALITIES}
    within_window = {mod: [] for mod in MODALITIES}
    for (subject, trial), rows in df.groupby(["subject", "trial"], sort=True):
        label, pooled, spread = read_trial(root, rows)
        trial_records.append((str(subject), int(trial), label, len(rows)))
        for mod in MODALITIES:
            trial_features[mod].append(pooled[mod])
            within_window[mod].append(spread[mod])

    table = pd.DataFrame(trial_records, columns=["subject", "trial", "label", "windows"])
    if not (table["windows"] == 11).all():
        print("WARNING: non-11-window trials:", table[table.windows != 11].to_dict("records")[:10])
    print("\nFEATURE VARIABILITY (pooled trial vectors)")
    for mod in MODALITIES:
        data = np.stack(trial_features[mod])
        between = float(data.std(axis=0).mean())
        inside = float(np.mean(within_window[mod]))
        print(f"{mod:5s} shape={data.shape}, between-trial std={between:.6g}, "
              f"within-trial window std={inside:.6g}, "
              f"near-constant dimensions={(data.std(axis=0) < 1e-8).sum()}/{data.shape[1]}")

    subject_names = set(table["subject"])
    if a.val_subject not in subject_names or a.test_subject not in subject_names:
        raise ValueError("Validation/test subjects not in manifest")
    if a.val_subject == a.test_subject:
        raise ValueError("Validation and test subject must differ")

    is_train = ~table.subject.isin([a.val_subject, a.test_subject]).to_numpy()
    is_val = (table.subject == a.val_subject).to_numpy()
    y = table.label.to_numpy(dtype=np.int64)
    print("\nTRIAL-LEVEL LINEAR PROBES (fit on training subjects only)")
    print(f"Train: {int(is_train.sum())} trials, Val: {int(is_val.sum())} trials, "
          f"Test: {(~is_train & ~is_val).sum()} trials (kept untouched)")

    majority = int(np.bincount(y[is_train], minlength=2).argmax())
    print("TRAIN MAJORITY -> VAL:", metric_line(y[is_val], np.full(is_val.sum(), majority)))
    for mod in (*MODALITIES, "concatenated"):
        if mod == "concatenated":
            x = np.concatenate([np.stack(trial_features[m]) for m in MODALITIES], axis=1)
        else:
            x = np.stack(trial_features[mod])
        # Standardization is fit ONLY on the 20 training subjects.
        clf = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=a.alpha, solver="lsqr", class_weight="balanced"),
        )
        clf.fit(x[is_train], y[is_train])
        train_pred = clf.predict(x[is_train])
        val_pred = clf.predict(x[is_val])
        print(f"{mod:12s} TRAIN {metric_line(y[is_train], train_pred)}")
        print(f"{'':12s} VALID {metric_line(y[is_val], val_pred)}")


if __name__ == "__main__":
    main()
