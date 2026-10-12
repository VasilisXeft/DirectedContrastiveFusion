"""Paired, label-controlled EEGPT input preprocessing benchmark on DEAP.

Three candidate representations for EXACTLY the same trial/window selection:
  cached_256_1e-6   : existing EEGPT [4,512] tokens from frozen 5-modality cache;
  eegpt_250_1e-6   : new EEGPT at 250 Hz, legacy amplitude scaling;
  eegpt_250_1e-3   : new EEGPT at 250 Hz, candidate millivolt scaling.

All are pooled into one 512-D vector per DEAP *trial*. A fixed RidgeClassifier
is assessed by development-only subject-independent LOSO. The reserved subject
(default s22) is never opened, fitted or evaluated.

To keep GPU cost reasonable, by default use 3 fixed 10-second windows per
60-second trial, at 0, 25, 50 seconds. The SAME windows are used for all
three representations. Runs can resume from complete subject-level caches.

Examples:
  python scripts/compare_deap_eegpt_inputs.py --deap-dir "C:/DEAP/data_preprocessed_python" --dry-run
  python scripts/compare_deap_eegpt_inputs.py --deap-dir "C:/DEAP/data_preprocessed_python" --subjects s01 s02 s03
  python scripts/compare_deap_eegpt_inputs.py --deap-dir "C:/DEAP/data_preprocessed_python"

This is a development-only diagnostic, not final unbiased model evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VARIANTS = ("cached_256_1e-6", "eegpt_250_1e-6", "eegpt_250_1e-3")


def window_starts(n_windows):
    """Start samples of cached 10-second/5-second-stride DEAP windows."""
    if n_windows not in (3, 6, 11):
        raise ValueError("windows-per-trial must be 3, 6 or 11")
    starts = np.linspace(0, 50 * 128, n_windows).round().astype(int)
    if not np.all(starts % (5 * 128) == 0):
        raise AssertionError(f"Window starts do not match existing cache: {starts}")
    return starts


def subject_trials(manifest, subject):
    groups = manifest.loc[manifest["subject"] == subject].groupby("trial", sort=True)
    result = []
    for trial, rows in groups:
        labels = rows["label"].unique()
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels for {subject} trial {trial}")
        result.append((int(trial), int(labels[0]), set(rows["file"].astype(str))))
    if not result:
        raise ValueError(f"No cached trials available for {subject}")
    return result


def contract(subject, trials, starts, checkpoint):
    references = [
        [subject, trial, label,
         [f"{subject}_trial{trial:02d}_start{start:06d}.npz" for start in starts]]
        for trial, label, _ in trials
    ]
    return {
        "format": "eegpt_trial_scale_probe_v1",
        "subject": subject,
        "checkpoint": checkpoint,
        "target_rate": 250,
        "source_rate": 128,
        "window_starts": starts.tolist(),
        "source_digest": hashlib.sha256(
            json.dumps(references, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "variants": VARIANTS,
    }


def load_cached_subject(path, spec, trial_count):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        actual = json.loads(str(z["contract_json"]))
        if actual != json.loads(json.dumps(spec)):
            raise ValueError(
                f"Incompatible existing subject benchmark cache {path}. "
                f"Do not mix protocols; choose a new --out directory."
            )
        result = {variant: np.asarray(z[variant], np.float32) for variant in VARIANTS}
        result["trials"] = np.asarray(z["trials"], np.int64)
        result["labels"] = np.asarray(z["labels"], np.int64)
        for name in VARIANTS:
            if result[name].shape != (trial_count, 512):
                raise ValueError(f"Wrong {name} cached shape for {path}: {result[name].shape}")
            if not np.isfinite(result[name]).all():
                raise ValueError(f"Non-finite cached features in {path}: {name}")
        return result


def extract_subject(subject, trial_entries, starts, deap_dir, cache_dir, encoder):
    raw_path = Path(deap_dir) / f"{subject}.dat"
    with raw_path.open("rb") as source:
        deap = pickle.load(source, encoding="latin1")
    raw, labels = np.asarray(deap["data"]), np.asarray(deap["labels"])
    output = {variant: [] for variant in VARIANTS}
    output["trials"], output["labels"] = [], []
    print(f"Extracting {subject}: {len(trial_entries)} labeled trials", flush=True)
    for i, (trial, label, available_filenames) in enumerate(trial_entries, 1):
        if int(labels[trial, 0] > 5.0) != label:
            raise ValueError(f"DEAP label != manifest label for {subject} trial {trial}")
        signal = np.asarray(raw[trial, :32, 3 * 128:], dtype=np.float32)
        if signal.shape != (32, 60 * 128):
            raise ValueError(f"Unexpected EEG shape for {subject} trial {trial}: {signal.shape}")

        cached = []
        for start in starts:
            filename = f"{subject}_trial{trial:02d}_start{start:06d}.npz"
            if filename not in available_filenames:
                raise ValueError(f"Trial {subject}/{trial} missing selected window {filename}")
            with np.load(Path(cache_dir) / filename, allow_pickle=False) as z:
                item = np.asarray(z["x__eeg"], dtype=np.float32)
                if item.shape != (4, 512) or not np.isfinite(item).all():
                    raise ValueError(f"Bad legacy EEGPT features in {filename}: {item.shape}")
                if int(np.asarray(z["target"]).item()) != label:
                    raise ValueError(f"Cached target mismatch for {filename}")
                cached.append(item.mean(axis=0))
        output[VARIANTS[0]].append(np.stack(cached).mean(axis=0))

        # Same EEG windows and same pretrained weights; only scale differs.
        for variant, scale in [(VARIANTS[1], 1e-6), (VARIANTS[2], 1e-3)]:
            encoder.input_scale = scale
            vectors = []
            for start in starts:
                window = signal[:, start:start + 10 * 128]
                item = encoder(window)
                if item.shape != (4, 512) or not np.isfinite(item).all():
                    raise ValueError(f"Bad EEGPT output {variant}, {subject}/{trial}")
                vectors.append(item.mean(axis=0))
            output[variant].append(np.stack(vectors).mean(axis=0))

        output["trials"].append(trial)
        output["labels"].append(label)
        if i % 5 == 0 or i == len(trial_entries):
            print(f"  {subject} trial {i}/{len(trial_entries)}", flush=True)

    result = {
        **{variant: np.stack(output[variant]).astype(np.float32) for variant in VARIANTS},
        "trials": np.asarray(output["trials"], dtype=np.int64),
        "labels": np.asarray(output["labels"], dtype=np.int64),
    }
    return result


def evaluate_loso(subjects, per_subject, alpha):
    trial_subjects = []
    targets = []
    embeddings = {name: [] for name in VARIANTS}
    for subject in subjects:
        item = per_subject[subject]
        n_trials = len(item["labels"])
        trial_subjects.extend([subject] * n_trials)
        targets.extend(item["labels"].tolist())
        for name in VARIANTS:
            embeddings[name].append(item[name])
    group = np.asarray(trial_subjects)
    y = np.asarray(targets, np.int64)
    embeddings = {name: np.concatenate(items, axis=0) for name, items in embeddings.items()}

    rows = []
    for subject in subjects:
        train = group != subject
        valid = group == subject
        for name in VARIANTS:
            clf = make_pipeline(
                StandardScaler(),
                RidgeClassifier(alpha=alpha, solver="lsqr", class_weight="balanced"),
            )
            clf.fit(embeddings[name][train], y[train])
            prediction = clf.predict(embeddings[name][valid])
            rows.append({
                "heldout_subject": subject,
                "representation": name,
                "train_trials": int(train.sum()),
                "validation_trials": int(valid.sum()),
                "accuracy": float(accuracy_score(y[valid], prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(y[valid], prediction)),
                "macro_f1": float(f1_score(y[valid], prediction, average="macro", zero_division=0)),
                "positive_predictions": int((prediction == 1).sum()),
            })
        scores = {row["representation"]: row["balanced_accuracy"]
                  for row in rows if row["heldout_subject"] == subject}
        print(f"{subject}: " + " ".join(f"{key}={scores[key]:.3f}" for key in VARIANTS), flush=True)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deap-dir", required=True)
    p.add_argument("--cache", default="data/features/deap_foundation_5m")
    p.add_argument("--reserve", default="s22")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subset of development subjects for a quick smoke test")
    p.add_argument("--out", default="runs/deap_foundation_5m/eegpt_input_comparison")
    p.add_argument("--windows-per-trial", type=int, default=3, choices=[3, 6, 11])
    p.add_argument("--alpha", type=float, default=100.)
    p.add_argument("--device", default=None)
    p.add_argument("--model-id", default="braindecode/eegpt-pretrained")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    manifest = pd.read_csv(Path(args.cache) / "manifest.csv", dtype={"subject": str})
    if not {"file", "subject", "trial", "label"}.issubset(manifest.columns):
        raise ValueError("Manifest is missing required file/subject/trial/label columns")
    if manifest["file"].duplicated().any():
        raise ValueError("Duplicate filenames in cached manifest")
    all_subjects = set(manifest["subject"].unique())
    if args.reserve not in all_subjects:
        raise ValueError("Reserved subject is not in the manifest")
    subjects = sorted(all_subjects - {args.reserve}) if args.subjects is None else sorted(set(args.subjects))
    if not set(subjects).issubset(all_subjects - {args.reserve}):
        raise ValueError("Selected subjects must exist and exclude final reserved subject")
    if len(subjects) < 3:
        raise ValueError("LOSO comparison requires at least 3 development subjects")

    # DROP reserved subject metadata before forming extraction groups.
    manifest = manifest.loc[manifest["subject"].isin(subjects)].copy()
    starts = window_starts(args.windows_per_trial)
    entries = {subject: subject_trials(manifest, subject) for subject in subjects}
    print(f"Reserved subject {args.reserve} excluded. "
          f"{len(subjects)} development subjects, "
          f"{sum(len(e) for e in entries.values())} trials, "
          f"{len(starts)} windows per trial.")
    print(f"Approx. pretrained EEGPT calls: "
          f"{sum(len(e) for e in entries.values()) * len(starts) * 4 * 2:,} "
          "(minus any resumed subject caches).")
    if args.dry_run:
        print("DRY RUN ONLY; no pretrained checkpoint or DEAP raw files loaded.")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_subject = {}
    encoder = None
    for subject in subjects:
        spec = contract(subject, entries[subject], starts, args.model_id)
        destination = out / f"{subject}.npz"
        result = load_cached_subject(destination, spec, len(entries[subject]))
        if result is None:
            if encoder is None:
                import torch
                from cmf.models.deap_eegpt import EEGPTFeatures
                device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
                encoder = EEGPTFeatures(device=device, model_id=args.model_id,
                                        target_sfreq=250, input_scale=1e-6)
            result = extract_subject(
                subject, entries[subject], starts,
                args.deap_dir, args.cache, encoder
            )
            tmp = out / f"{subject}.tmp.npz"
            np.savez_compressed(tmp, **result, contract_json=np.asarray(json.dumps(spec)))
            os.replace(tmp, destination)
            print(f"Saved {destination}", flush=True)
        else:
            print(f"Reusing {destination}", flush=True)
        expected_trials = np.asarray([trial for trial, _, _ in entries[subject]])
        expected_labels = np.asarray([label for _, label, _ in entries[subject]])
        if not np.array_equal(result["trials"], expected_trials) or not np.array_equal(
            result["labels"], expected_labels
        ):
            raise ValueError(f"Trial-label order mismatch in saved subject cache: {subject}")
        per_subject[subject] = result

    print("\nPAIRED DEVELOPMENT-ONLY LOSO (all representations use same trials)")
    metrics = evaluate_loso(subjects, per_subject, args.alpha)
    metrics_path = out / "dev_loso_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = metrics.groupby("representation").agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_accuracy=("accuracy", "mean"),
    ).loc[list(VARIANTS)]
    print("\n" + summary.round(4).to_string())
    pivot = metrics.pivot(index="heldout_subject", columns="representation",
                          values="balanced_accuracy")
    delta_scale = pivot["eegpt_250_1e-3"] - pivot["eegpt_250_1e-6"]
    delta_rate = pivot["eegpt_250_1e-6"] - pivot["cached_256_1e-6"]
    print(f"\nPaired mean balanced-accuracy delta for amplitude scaling "
          f"(250Hz, 1e-3 minus 1e-6): {delta_scale.mean():+.4f} "
          f"(wins/ties/losses={(delta_scale > 0).sum()}/{(delta_scale == 0).sum()}/{(delta_scale < 0).sum()})")
    print(f"Paired mean balanced-accuracy delta for sample rate "
          f"(250Hz minus cached 256Hz, scale 1e-6): {delta_rate.mean():+.4f}")
    print(f"Saved {metrics_path}. Reserved subject {args.reserve} never evaluated.")
    print("Exploratory development folds; do not treat results as final nested-CV scores.")


if __name__ == "__main__":
    main()
