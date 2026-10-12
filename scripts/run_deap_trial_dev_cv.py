"""Seven disjoint 18-train/3-validation development folds, reserving s22.

Checkpoint selection uses the three validation subjects. These are development
metrics, NOT an unbiased test estimate. Models and training-only scalers are
independent in each fold. Re-runs skip completed models.

Examples:
  python scripts/run_deap_trial_dev_cv.py --dry-run
  python scripts/run_deap_trial_dev_cv.py --groups 0 1 --modes ridge_all unimodal_eeg --seeds 42
  python scripts/run_deap_trial_dev_cv.py --modes ridge_all unimodal_eeg early directed_topk --seeds 42 --epochs 15 --patience 5 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_deap_trial_fusion import development_validation_groups, expand_modes


def gather(root, groups, seeds, modes):
    group_rows = []
    subject_rows = []
    for group in groups:
        for seed in seeds:
            for mode in modes:
                path = root / f"dev_group_{group:02d}" / f"seed{seed}" / mode / "validation_metrics.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing completed model: {path}")
                met = json.loads(path.read_text(encoding="utf-8"))
                group_rows.append({
                    "group": group, "seed": seed, "mode": mode,
                    "balanced_accuracy": met["subject_mean"]["balanced_accuracy"],
                    "macro_f1": met["subject_mean"]["macro_f1"],
                    "pooled_bacc": met["pooled"]["balanced_accuracy"],
                    "pooled_macro_f1": met["pooled"]["macro_f1"],
                })
                for subject, entry in met["per_subject"].items():
                    subject_rows.append({
                        "group": group, "seed": seed, "mode": mode,
                        "subject": subject, "n_trials": entry["trials"],
                        "balanced_accuracy": entry["balanced_accuracy"],
                        "macro_f1": entry["macro_f1"],
                        "accuracy": entry["accuracy"],
                        "pred_class0": entry["predicted_class_counts"][0],
                        "pred_class1": entry["predicted_class_counts"][1],
                    })
    gr = pd.DataFrame(group_rows)
    sr = pd.DataFrame(subject_rows)
    if sr.duplicated(["seed", "mode", "subject"]).any():
        raise RuntimeError("Repeated validation subject across CV groups")
    if len(sr) != len(groups) * len(seeds) * len(modes) * 3:
        raise RuntimeError("Missing validation subjects in CV output")
    summary = sr.assign(
        one_class=(sr.pred_class0 == 0) | (sr.pred_class1 == 0)
    ).groupby("mode", sort=False).agg(
        subject_folds=("subject", "count"),
        mean_bacc=("balanced_accuracy", "mean"),
        sd_bacc=("balanced_accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        sd_macro_f1=("macro_f1", "std"),
        one_class_folds=("one_class", "sum"),
    ).sort_values(["mean_macro_f1", "mean_bacc"], ascending=False)
    return gr, sr, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/features/deap_foundation_5m")
    parser.add_argument("--out", default="runs/deap_foundation_5m/trial_fusion_dev_cv")
    parser.add_argument("--reserve", default="s22")
    parser.add_argument("--groups", nargs="+", type=int, default=list(range(7)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--modes", nargs="+", default=[
        "ridge_all", "unimodal_eeg", "early", "directed_topk"
    ])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    a = parser.parse_args()
    if len(set(a.groups)) != len(a.groups) or not a.groups or not a.seeds:
        parser.error("Group IDs must be unique and seed/group lists nonempty")
    if any(i not in range(7) for i in a.groups):
        parser.error("Group IDs must range from 0 to 6")
    modes = expand_modes(a.modes)
    manifest = pd.read_csv(Path(a.cache) / "manifest.csv", dtype={"subject": str})
    groups = development_validation_groups(manifest, a.reserve)
    print("Reserved subject:", a.reserve, "(never loaded by benchmark)")
    for i in a.groups:
        train = set(manifest["subject"].unique()) - {a.reserve, *groups[i]}
        if len(train) != 18:
            raise RuntimeError("Expected exactly 18 training subjects")
        print(f"group {i}: val={groups[i]}, train={len(train)} subjects")
    print(f"Requested {len(a.groups)} groups x {len(a.seeds)} seeds x "
          f"{len(modes)} models = {len(a.groups)*len(a.seeds)*len(modes)} runs.")
    if a.dry_run:
        return

    if not a.aggregate_only:
        script = Path(__file__).with_name("run_deap_trial_fusion.py")
        for i in a.groups:
            for seed in a.seeds:
                cmd = [
                    sys.executable, str(script), "--cache", a.cache,
                    "--out", a.out, "--reserve", a.reserve,
                    "--val-group", str(i), "--seed", str(seed),
                    "--modes", *a.modes, "--epochs", str(a.epochs),
                    "--patience", str(a.patience),
                    "--batch-size", str(a.batch_size),
                    "--lr", str(a.lr), "--weight-decay", str(a.weight_decay),
                    "--topk", str(a.topk), "--d-model", str(a.d_model),
                    "--heads", str(a.heads), "--skip-existing",
                ]
                if a.cpu:
                    cmd.append("--cpu")
                print(f"\n== GROUP {i} SEED {seed} ==", flush=True)
                subprocess.run(cmd, check=True)

    root = Path(a.out)
    fold, subject, summary = gather(root, a.groups, a.seeds, modes)
    root.mkdir(parents=True, exist_ok=True)
    fold.to_csv(root / "fold_metrics.csv", index=False)
    subject.to_csv(root / "subject_metrics.csv", index=False)
    summary.to_csv(root / "summary.csv")
    print("\nSEVEN-GROUP DEVELOPMENT LEADERBOARD")
    print(summary.round(4).to_string())
    print(f"\nSaved: {root / 'summary.csv'}")
    print("These validation metrics are used for checkpoint selection and "
          "are not unbiased test estimates. s22 was never evaluated.")


if __name__ == "__main__":
    main()
