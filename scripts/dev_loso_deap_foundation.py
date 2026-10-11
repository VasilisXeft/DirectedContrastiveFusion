"""Development-only subject-independent probes for cached DEAP foundation features.

Reserves a final subject entirely (default s22) and runs 21 development folds:
one validation subject, all other development subjects for training.

Usage:
    python scripts/dev_loso_deap_foundation.py
    python scripts/dev_loso_deap_foundation.py --alpha 100 --reserve s22

A fixed alpha is used in every fold; tune hyperparameters inside nested
subject-grouped validation before reporting an unbiased final performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from diagnose_deap_foundation import MODALITIES, read_trial


def score(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "pred_0": int((y_pred == 0).sum()),
        "pred_1": int((y_pred == 1).sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/features/deap_foundation_5m")
    p.add_argument("--reserve", default="s22", help="Keep this subject out of ALL development folds")
    p.add_argument("--alpha", type=float, default=100.0)
    p.add_argument("--out", default="runs/deap_foundation_5m/dev_loso_linear_probes.csv")
    args = p.parse_args()

    root = Path(args.cache)
    manifest = pd.read_csv(root / "manifest.csv", dtype={"subject": str})
    required = {"subject", "trial", "label", "file"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Missing manifest columns: {required - set(manifest.columns)}")
    if args.reserve not in set(manifest["subject"]):
        raise ValueError(f"Reserved subject {args.reserve} is not present in the manifest")
    # Crucial: do not even load the reserved subject's features or labels.
    manifest = manifest.loc[manifest.subject != args.reserve].copy()
    if manifest["file"].duplicated().any():
        raise ValueError("Duplicate filenames in development manifest")
    missing = [f for f in manifest["file"] if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cached development embeddings, first: {missing[:3]}")

    meta = []
    feature_lists = {mod: [] for mod in MODALITIES}
    groups = manifest.groupby(["subject", "trial"], sort=True)
    print(f"Loading {groups.ngroups} development trials; reserved subject {args.reserve} untouched.", flush=True)
    for i, ((subject, trial), rows) in enumerate(groups, start=1):
        label, pooled, _ = read_trial(root, rows)
        meta.append((str(subject), int(trial), int(label)))
        for mod in MODALITIES:
            feature_lists[mod].append(pooled[mod])
        if i % 100 == 0:
            print(f"Loaded {i} trials", flush=True)

    trials = pd.DataFrame(meta, columns=["subject", "trial", "label"])
    xs = {mod: np.stack(v).astype(np.float32) for mod, v in feature_lists.items()}
    xs["concatenated"] = np.concatenate([xs[mod] for mod in MODALITIES], axis=1)
    y = trials.label.to_numpy(dtype=np.int64)
    subjects = sorted(trials.subject.unique())
    if len(subjects) < 3:
        raise ValueError("At least three development subjects required")

    print(f"\nEvaluating {len(subjects)} development LOSO folds, fixed alpha={args.alpha:g}", flush=True)
    rows = []
    for fold, heldout in enumerate(subjects, 1):
        train = (trials.subject != heldout).to_numpy()
        val = ~train
        train_labels, val_labels = y[train], y[val]
        if len(set(train_labels)) != 2:
            raise ValueError(f"Training labels lack a class in fold {heldout}")

        for modality, x in xs.items():
            clf = make_pipeline(
                StandardScaler(),
                RidgeClassifier(alpha=args.alpha, solver="lsqr", class_weight="balanced"),
            )
            clf.fit(x[train], train_labels)
            prediction = clf.predict(x[val])
            metrics = score(val_labels, prediction)
            rows.append({
                "heldout_subject": heldout, "modality": modality,
                "n_train_trials": int(train.sum()), "n_val_trials": int(val.sum()),
                "n_val_neg": int((val_labels == 0).sum()),
                "n_val_pos": int((val_labels == 1).sum()),
                **metrics,
            })
        fold_scores = {r["modality"]: r["balanced_accuracy"]
                       for r in rows if r["heldout_subject"] == heldout}
        print(f"[{fold:02d}/{len(subjects)}] {heldout}: " +
              " ".join(f"{m}={fold_scores[m]:.3f}" for m in xs), flush=True)

    result = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print("\nDEVELOPMENT LOSO SUMMARY (subject-weighted, heldout subject is NOT tested)")
    summary = result.groupby("modality", sort=False).agg(
        mean_bacc=("balanced_accuracy", "mean"),
        std_bacc=("balanced_accuracy", "std"),
        median_bacc=("balanced_accuracy", "median"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_acc=("accuracy", "mean"),
        above_chance_folds=("balanced_accuracy", lambda x: int((x > 0.5).sum())),
    ).sort_values("mean_bacc", ascending=False)
    print(summary.round(3).to_string())
    print(f"\nSaved all {len(result)} fold results to {out}")
    print(f"Reserved {args.reserve} for eventual evaluation after model selection.")
    print("These are exploratory development scores, NOT final nested-CV estimates.")


if __name__ == "__main__":
    main()
