from pathlib import Path
import json
import numpy as np
import pandas as pd

from cmf.config import load_config
from cmf.train import train_from_config
from cmf.pretrain import pretrain_fold
from cmf.utils.io import ensure_dir


def _subjects(manifest, subject_column):
    if subject_column not in manifest.columns:
        raise ValueError(f"LOSO subject column '{subject_column}' not found in manifest.csv")
    return sorted(manifest[subject_column].astype(str).unique().tolist())


def _fold_manifests(manifest, test_subject, val_subject, subject_column):
    base = manifest.copy()
    base[subject_column] = base[subject_column].astype(str)
    train = base[~base[subject_column].isin([test_subject, val_subject])].copy()
    val = base[base[subject_column] == val_subject].copy()
    test = base[base[subject_column] == test_subject].copy()
    for frame, split in ((train, "train"), (val, "val"), (test, "test")):
        frame["split"] = split
    return {"train": train, "val": val, "test": test}


def run_loso(config_path, subject=None, run_all=False, mode=None):
    cfg = load_config(config_path)
    dcfg = cfg["dataset"]
    lcfg = cfg.get("loso", {})
    if not lcfg.get("enabled", True):
        raise ValueError("LOSO is disabled in this config.")

    cache = Path(dcfg["cache_dir"])
    manifest = pd.read_csv(cache / "manifest.csv")
    subject_column = lcfg.get("subject_column", "subject")
    subjects = _subjects(manifest, subject_column)
    if len(subjects) < 3:
        raise ValueError("LOSO requires at least 3 subjects for train/validation/test.")

    requested = subjects if run_all else [str(subject or lcfg.get("test_subject", subjects[-1]))]
    unknown = [s for s in requested if s not in subjects]
    if unknown:
        raise ValueError(f"Unknown LOSO subject(s): {unknown}. Available: {subjects}")

    val_strategy = lcfg.get("validation_strategy", "previous_subject")
    fixed_val = lcfg.get("validation_subject")
    results = []
    for test_subject in requested:
        if fixed_val is not None:
            val_subject = str(fixed_val)
            if val_subject == test_subject:
                raise ValueError("validation_subject cannot equal the held-out test subject.")
        elif val_strategy == "previous_subject":
            i = subjects.index(test_subject)
            val_subject = subjects[(i - 1) % len(subjects)]
        elif val_strategy == "next_subject":
            i = subjects.index(test_subject)
            val_subject = subjects[(i + 1) % len(subjects)]
        else:
            raise ValueError("validation_strategy must be previous_subject or next_subject, or set validation_subject.")

        splits = _fold_manifests(manifest, test_subject, val_subject, subject_column)
        print(f"LOSO fold: test={test_subject} val={val_subject} train_subjects={len(subjects)-2} "
              f"samples(train/val/test)={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
        encoder_checkpoints = None
        if cfg.get("pretraining", {}).get("enabled", False):
            pre = pretrain_fold(config_path, splits, test_subject, val_subject)
            encoder_checkpoints = {m: info["checkpoint"] for m, info in pre.items()}
        res = train_from_config(config_path, mode=mode, split_manifests=splits,
                                run_name=f"test_{test_subject}_val_{val_subject}", return_metrics=True,
                                encoder_checkpoints=encoder_checkpoints)
        results.append({"test_subject": test_subject, "val_subject": val_subject, **res})

    if run_all:
        task = dcfg.get("task", "classification")
        summary = {"config": str(config_path), "mode": mode or cfg["model"].get("mode"), "folds": results}
        if task == "classification":
            for key in ("accuracy", "macro_f1"):
                vals = [r["test"][key] for r in results]
                summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
        else:
            vals = [r["test"]["rmse"] for r in results]
            summary["rmse"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
        out = ensure_dir(Path(cfg.get("output_dir", "runs")) / dcfg["name"] / "loso")
        (out / f"{mode or cfg['model'].get('mode','model')}_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k:v for k,v in summary.items() if k != "folds"}, indent=2))
    return results
