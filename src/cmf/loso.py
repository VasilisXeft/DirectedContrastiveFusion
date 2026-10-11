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


def _validation_subjects(subjects, test_subject, lcfg):
    """Pick validation subjects deterministically, without consulting labels.

    For 22 subjects and spaced_previous/count=3, offsets 1, 8, 15
    place validation participants across the cohort.
    """
    count = int(lcfg.get("validation_subject_count", 1))
    if not 1 <= count <= len(subjects) - 2:
        raise ValueError(
            f"validation_subject_count must be between 1 and {len(subjects)-2}; got {count}"
        )

    fixed = lcfg.get("validation_subjects", lcfg.get("validation_subject"))
    if fixed is not None:
        chosen = [str(v) for v in fixed] if isinstance(fixed, (list, tuple)) else [str(fixed)]
        if len(chosen) != count:
            raise ValueError(
                f"Fixed validation_subjects has {len(chosen)} entries; expected {count}"
            )
    else:
        strategy = lcfg.get("validation_strategy", "previous_subject")
        idx = subjects.index(test_subject)
        n = len(subjects)
        if strategy == "previous_subject":
            offsets = list(range(1, count + 1))
            chosen = [subjects[(idx - offset) % n] for offset in offsets]
        elif strategy == "next_subject":
            offsets = list(range(1, count + 1))
            chosen = [subjects[(idx + offset) % n] for offset in offsets]
        elif strategy == "spaced_previous":
            offsets = [1 + (i * (n - 1)) // count for i in range(count)]
            chosen = [subjects[(idx - offset) % n] for offset in offsets]
        else:
            raise ValueError(
                "validation_strategy must be previous_subject, next_subject, "
                "or spaced_previous (or set validation_subjects)"
            )

    if len(set(chosen)) != count or test_subject in chosen:
        raise ValueError(f"Validation subjects must be unique and distinct from test={test_subject}: {chosen}")
    unknown = set(chosen) - set(subjects)
    if unknown:
        raise ValueError(f"Unknown validation subject(s): {sorted(unknown)}")
    return chosen


def _fold_manifests(manifest, test_subject, val_subjects, subject_column):
    base = manifest.copy()
    base[subject_column] = base[subject_column].astype(str)
    val_subjects = [val_subjects] if isinstance(val_subjects, str) else list(val_subjects)
    held_out = set(val_subjects) | {test_subject}
    train = base[~base[subject_column].isin(held_out)].copy()
    val = base[base[subject_column].isin(val_subjects)].copy()
    test = base[base[subject_column] == test_subject].copy()
    for frame, split in ((train, "train"), (val, "val"), (test, "test")):
        frame["split"] = split
        if frame.empty:
            raise ValueError(f"No samples assigned to {split} fold")
    if sum(map(len, (train, val, test))) != len(base):
        raise RuntimeError("LOSO fold failed to partition the manifest")
    return {"train": train, "val": val, "test": test}


def run_loso(config_path, subject=None, run_all=False, mode=None, topk=None, seed=None, experiment_name=None):
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

    results = []
    for test_subject in requested:
        val_subjects = _validation_subjects(subjects, test_subject, lcfg)
        splits = _fold_manifests(manifest, test_subject, val_subjects, subject_column)
        actual_train = sorted(set(splits["train"][subject_column].astype(str)))
        actual_val = sorted(set(splits["val"][subject_column].astype(str)))
        actual_test = sorted(set(splits["test"][subject_column].astype(str)))
        if (len(actual_train) != len(subjects)-1-len(val_subjects)
                or len(actual_val) != len(val_subjects)
                or actual_test != [test_subject]
                or (set(actual_train) & (set(actual_val) | set(actual_test)))):
            raise RuntimeError("Invalid LOSO participant partition")
        val_tag = "-".join(val_subjects)
        print(f"LOSO fold: test={test_subject} val={val_subjects} "
              f"train_subjects={len(actual_train)} val_subjects={len(actual_val)} "
              f"samples(train/val/test)={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
        encoder_checkpoints = None
        if cfg.get("pretraining", {}).get("enabled", False):
            pre = pretrain_fold(config_path, splits, test_subject, val_tag)
            encoder_checkpoints = {m: info["checkpoint"] for m, info in pre.items()}
        res = train_from_config(config_path, mode=mode, split_manifests=splits,
                                run_name=(f"{experiment_name}/" if experiment_name else "") + f"test_{test_subject}_val_{val_tag}", return_metrics=True,
                                encoder_checkpoints=encoder_checkpoints, topk=topk, seed=seed)
        results.append({"test_subject": test_subject, "val_subjects": val_subjects, "val_subject": val_subjects[0] if len(val_subjects) == 1 else val_tag, **res})

    if run_all:
        task = dcfg.get("task", "classification")
        summary = {"config": str(config_path), "mode": mode or cfg["model"].get("mode"), "topk": topk, "seed": seed, "experiment_name": experiment_name, "folds": results}
        if task == "classification":
            for key in ("accuracy", "balanced_accuracy", "macro_f1"):
                vals = [r["test"][key] for r in results]
                summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
        else:
            vals = [r["test"]["rmse"] for r in results]
            summary["rmse"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
        out = ensure_dir(Path(cfg.get("output_dir", "runs")) / dcfg["name"] / "loso")
        (out / f"{experiment_name or (mode or cfg['model'].get('mode','model'))}_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k:v for k,v in summary.items() if k != "folds"}, indent=2))
    return results
