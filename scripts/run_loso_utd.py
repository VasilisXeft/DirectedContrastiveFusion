import argparse
import copy
import csv
from pathlib import Path
import yaml
import pandas as pd

from cmf.config import load_config
from cmf.datasets.preprocess_utd_mhad import preprocess_utd_mhad
from cmf.train import train_from_config
from cmf.evaluate import evaluate_checkpoint


def main():
    p = argparse.ArgumentParser(description="LOSO-style UTD-MHAD experiment with a separate validation subject.")
    p.add_argument("--config", default="configs/utd_mhad_subset.yaml")
    p.add_argument("--modes", nargs="+", default=["full", "contrastive_topk"])
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 9)))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-root", default="runs/utd_mhad_loso")
    p.add_argument("--cache-root", default="data/processed/utd_mhad_loso")
    args = p.parse_args()

    base = load_config(args.config)
    rows = []
    tmp_dir = Path(args.output_root) / "configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects
    for i, test_subject in enumerate(subjects):
        # Nested subject-wise validation: the next subject is validation, all remaining are training.
        val_subject = subjects[(i + 1) % len(subjects)]
        train_subjects = [s for s in subjects if s not in {test_subject, val_subject}]

        cfg = copy.deepcopy(base)
        cfg["dataset"]["train_subjects"] = train_subjects
        cfg["dataset"]["val_subjects"] = [val_subject]
        cfg["dataset"]["test_subjects"] = [test_subject]
        cfg["dataset"]["cache_dir"] = str(Path(args.cache_root) / f"fold_{test_subject}")
        cfg["output_dir"] = str(Path(args.output_root) / f"fold_{test_subject}")
        if args.epochs is not None:
            cfg["training"]["epochs"] = args.epochs

        cfg_path = tmp_dir / f"fold_{test_subject}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False)

        print("\n" + "=" * 72)
        print(f"Fold test=S{test_subject} val=S{val_subject} train={train_subjects}")
        print("=" * 72)
        preprocess_utd_mhad(cfg)

        for mode in args.modes:
            outdir = train_from_config(cfg_path, mode=mode)
            ckpt = Path(outdir) / "best.pt"
            metrics = evaluate_checkpoint(ckpt, cfg["dataset"]["cache_dir"], "test")
            rec = {
                "test_subject": test_subject,
                "val_subject": val_subject,
                "mode": mode,
                **metrics,
                "checkpoint": str(ckpt),
            }
            rows.append(rec)
            print("TEST", rec)

        pd.DataFrame(rows).to_csv(Path(args.output_root) / "loso_results_partial.csv", index=False)

    df = pd.DataFrame(rows)
    out = Path(args.output_root)
    df.to_csv(out / "loso_all_results.csv", index=False)
    numeric = [c for c in ["loss", "accuracy", "macro_f1", "rmse"] if c in df.columns]
    summary = df.groupby("mode")[numeric].agg(["mean", "std"])
    summary.to_csv(out / "loso_summary.csv")
    print("\nFinal summary:\n", summary)


if __name__ == "__main__":
    main()
