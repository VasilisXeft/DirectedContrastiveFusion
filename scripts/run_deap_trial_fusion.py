"""Trial-level DEAP fusion benchmark using already extracted five-modality features.

Each of 834/874 development/all DEAP trials is ONE label/sample rather than
eleven overlapping supervised windows. The model sees 11 chronological tokens
per modality, obtained by pooling each window's 4 foundation-model tokens.

DEVELOPMENT ONLY: reserves s22 and never reads its NPZ files. With 22 total
subjects, validation subjects are s21, s14, s07 and training subjects are
the remaining 18. This is a development comparison, NOT an unbiased test
estimate: the validation subjects are used for checkpoint selection.

Examples:
 python scripts/run_deap_trial_fusion.py --dry-run
 python scripts/run_deap_trial_fusion.py --modes ridge_all
 python scripts/run_deap_trial_fusion.py --modes unimodal_eeg early directed_topk --epochs 15
 python scripts/run_deap_trial_fusion.py --modes full --batch-size 8

Each run records per-subject validation metrics, predictions, a checkpoint and
the training-only feature standardizer. Existing NPZs are never altered.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from cmf.datasets.common import collate_multimodal
from cmf.datasets.deap_trials import DEAPTrialDataset, MODALITY_ORDER, group_trials
from cmf.feature_normalization import FeatureStandardizer
from cmf.loso import _fold_manifests, _subjects, _validation_subjects
from cmf.models.fusion import ContrastiveSparseFusion
from cmf.utils.seed import seed_everything


NEURAL_MODES = {
    "unimodal_eeg", "unimodal_ppg", "unimodal_eda", "unimodal_temp",
    "unimodal_face", "late", "early", "full", "directed_topk",
}
RIDGE_MODES = {"ridge_" + m for m in MODALITY_ORDER} | {"ridge_concat"}
SUPPORTED_MODES = NEURAL_MODES | RIDGE_MODES | {"ridge_all"}
DEFAULT_MODES = ["ridge_all", "unimodal_eeg", "unimodal_eda",
                 "late", "early", "full", "directed_topk"]


def expand_modes(requested):
    result = []
    for mode in requested:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unknown mode {mode}. Available modes: {sorted(SUPPORTED_MODES)}")
        additions = (
            ["ridge_" + m for m in MODALITY_ORDER] + ["ridge_concat"]
            if mode == "ridge_all" else [mode]
        )
        for name in additions:
            if name not in result:
                result.append(name)
    return result


def metric_summary(truth, predictions, subjects):
    """Report subject-averaged metrics, pooled metrics and each confusion matrix."""
    truth = np.asarray(truth, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    subjects = np.asarray(subjects).astype(str)
    if len(truth) == 0 or len(truth) != len(predictions) or len(truth) != len(subjects):
        raise ValueError("Evaluation targets, predictions and subject IDs must align")
    if set(truth) - {0, 1} or set(predictions) - {0, 1}:
        raise ValueError("Only binary valence labels supported")
    per = {}
    for subject in sorted(set(subjects)):
        selected = subjects == subject
        y, p = truth[selected], predictions[selected]
        if set(y) != {0, 1}:
            raise ValueError(f"Validation subject {subject} is missing a target class")
        per[subject] = {
            "trials": int(selected.sum()),
            "true_class_counts": np.bincount(y, minlength=2).tolist(),
            "predicted_class_counts": np.bincount(p, minlength=2).tolist(),
            "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
        }
    pooled = {
        "trials": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "predicted_class_counts": np.bincount(predictions, minlength=2).tolist(),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
    }
    average = {
        name: float(np.mean([v[name] for v in per.values()]))
        for name in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    return {"subject_mean": average, "pooled": pooled, "per_subject": per}


def score_model(model, loader, device):
    model.eval()
    targets, predictions, subjects, trials, losses, pairs = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            mods = {m: x.to(device) for m, x in batch["modalities"].items()}
            present = {m: x.to(device) for m, x in batch["present"].items()}
            q = {m: x.to(device) for m, x in batch["quality_penalty"].items()}
            logits, aux = model(mods, present, q)
            y = batch["target"].long().to(device).view(-1)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite trial-level evaluation logits")
            losses.extend(F.cross_entropy(logits, y, reduction="none").cpu().tolist())
            targets.extend(y.cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            subjects.extend(item["subject"] for item in batch["meta"])
            trials.extend(int(item["trial"]) for item in batch["meta"])
            pairs.extend([float(aux["pair_count"].item())] * len(batch["meta"]))
    metrics = metric_summary(targets, predictions, subjects)
    metrics["cross_entropy"] = float(np.mean(losses))
    metrics["mean_selected_pairs"] = float(np.mean(pairs))
    frame = pd.DataFrame({
        "subject": subjects, "trial": trials,
        "true_label": targets, "predicted_label": predictions,
    })
    return metrics, frame


def ridge_predict(mode, train_ds, val_ds):
    def pooled_features(ds, which):
        blocks = []
        for _, tokens in ds.records:
            names = MODALITY_ORDER if which == "concat" else (which,)
            blocks.append(
                np.concatenate([tokens[name].mean(0).numpy() for name in names])
            )
        return np.stack(blocks).astype(np.float32)
    which = mode.removeprefix("ridge_")
    x_train = pooled_features(train_ds, which)
    x_val = pooled_features(val_ds, which)
    clf = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=100.0, solver="lsqr", class_weight="balanced"),
    )
    clf.fit(x_train, train_ds.labels())
    pred = clf.predict(x_val)
    return metric_summary(val_ds.labels(), pred, val_ds.subjects()), pd.DataFrame({
        "subject": val_ds.subjects(),
        "trial": [group["trial"] for group, _ in val_ds.records],
        "true_label": val_ds.labels(),
        "predicted_label": pred,
    })


def build_neural_model(mode, shapes, topk, d_model, heads):
    return ContrastiveSparseFusion(
        shapes, num_outputs=2, task="classification",
        mode=mode, topk=topk, d_model=d_model, heads=heads,
        encoder_configs={
            name: {"type": "precomputed", "feature_dim": shape[-1]}
            for name, shape in shapes.items()
        },
        # First benchmark isolates foundation-feature value + fusion architecture.
        # Reliability gating is a separate ablation, not silently enabled here.
        reliability=False,
        gumbel=False,
        selector_temperature=0.7,
        quality_lambda=0.0,
    )


def train_neural(mode, train_ds, val_ds, normalizer, args, outdir):
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model = build_neural_model(
        mode, train_ds.shapes(), args.topk, args.d_model, args.heads
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", collate_fn=collate_multimodal,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda", collate_fn=collate_multimodal,
    )
    labels = train_ds.labels()
    counts = np.bincount(labels, minlength=2)
    if not np.all(counts):
        raise ValueError(f"Train split missing class: {counts.tolist()}")
    class_weights = torch.tensor(
        len(labels) / (2.0 * counts), dtype=torch.float32, device=device
    )
    print(f"{mode}: {device}, train labels={counts.tolist()}, "
          f"train trials={len(train_ds)}, val trials={len(val_ds)}", flush=True)
    best_score = (-float("inf"), -float("inf"))
    bad_epochs = 0
    history = []
    best_metrics = None
    best_predictions = None
    outdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        tic = time.time()
        for batch in train_loader:
            mods = {m: x.to(device) for m, x in batch["modalities"].items()}
            present = {m: x.to(device) for m, x in batch["present"].items()}
            q = {m: x.to(device) for m, x in batch["quality_penalty"].items()}
            y = batch["target"].long().to(device).view(-1)
            logits, aux = model(mods, present, q)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in mode={mode} epoch={epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics, predictions = score_model(model, val_loader, device)
        current = (
            metrics["subject_mean"]["macro_f1"],
            metrics["subject_mean"]["balanced_accuracy"],
        )
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": metrics["cross_entropy"],
            "val_subject_mean_macro_f1": current[0],
            "val_subject_mean_bacc": current[1],
            "val_pooled_macro_f1": metrics["pooled"]["macro_f1"],
            "val_predicted_counts": metrics["pooled"]["predicted_class_counts"],
            "mean_selected_pairs": metrics["mean_selected_pairs"],
            "seconds": round(time.time() - tic, 2),
        }
        history.append(entry)
        print(json.dumps({"mode": mode, **entry}), flush=True)
        if current[0] > best_score[0] + 1e-5 or (
            abs(current[0] - best_score[0]) <= 1e-5 and
            current[1] > best_score[1] + 1e-5
        ):
            best_score = current
            bad_epochs = 0
            best_metrics = metrics
            best_predictions = predictions.copy()
            torch.save({
                "model": model.state_dict(),
                "mode": mode,
                "shapes": train_ds.shapes(),
                "feature_normalization": (
                    normalizer.state_dict() if normalizer is not None else None
                ),
                "seed": args.seed,
                "epoch": epoch,
                "best_validation": metrics,
                "settings": {
                    "topk": args.topk, "d_model": args.d_model,
                    "heads": args.heads, "reliability": False,
                    "window_tokens": 11,
                },
            }, outdir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping {mode} after {epoch} epochs", flush=True)
                break

    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    (outdir / "validation_metrics.json").write_text(
        json.dumps(best_metrics, indent=2)
    )
    best_predictions.to_csv(outdir / "validation_predictions.csv", index=False)
    return best_metrics


def development_validation_groups(manifest, reserved_subject="s22"):
    """Seven disjoint 3-subject groups among the 21 development participants.

    For s22 held out, group 6 is [s07, s14, s21], equal to the
    original three-subject validation set. All choices use IDs only.
    """
    subject_ids = _subjects(manifest, "subject")
    if reserved_subject not in subject_ids or len(subject_ids) != 22:
        raise ValueError("Expected exactly 22 subjects including reserved test subject")
    dev_ids = [subject for subject in subject_ids if subject != reserved_subject]
    if len(dev_ids) != 21:
        raise AssertionError("Expected 21 development subjects")
    groups = [[dev_ids[i + j * 7] for j in range(3)] for i in range(7)]
    seen = [subject for group in groups for subject in group]
    if len(seen) != 21 or set(seen) != set(dev_ids):
        raise AssertionError("Validation groups must partition development subjects")
    return groups


def split_development(manifest, reserved_subject="s22", val_group=None):
    """Use exact same 18/3/1 split as five-modal LOSO; NEVER return test rows."""
    subject_ids = _subjects(manifest, "subject")
    if reserved_subject not in subject_ids:
        raise ValueError(f"Reserved test subject {reserved_subject} not found")
    if len(subject_ids) != 22:
        raise ValueError(
            f"This specific 18/3/1 benchmark expects 22 subjects, found {len(subject_ids)}"
        )
    if val_group is None:
        val_ids = _validation_subjects(subject_ids, reserved_subject, {
            "validation_strategy": "spaced_previous",
            "validation_subject_count": 3,
        })
    else:
        groups = development_validation_groups(manifest, reserved_subject)
        if not 0 <= val_group < len(groups):
            raise ValueError(f"val_group must be between 0 and {len(groups)-1}")
        val_ids = groups[val_group]
    partitions = _fold_manifests(manifest, reserved_subject, val_ids, "subject")
    if partitions["train"]["subject"].nunique() != 18:
        raise AssertionError("Expected 18 distinct training subjects")
    if partitions["val"]["subject"].nunique() != 3:
        raise AssertionError("Expected 3 distinct validation subjects")
    if partitions["test"]["subject"].nunique() != 1:
        raise AssertionError("Expected exactly one reserved subject")
    if set(partitions["train"]["subject"]) & set(partitions["val"]["subject"]):
        raise AssertionError("Subject leakage between training and validation")
    # Important: caller must not inspect reserved NPZ files.
    return partitions["train"], partitions["val"], val_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/features/deap_foundation_5m")
    p.add_argument("--out", default="runs/deap_foundation_5m/trial_fusion_dev")
    p.add_argument("--reserve", default="s22")
    p.add_argument("--val-group", type=int, choices=range(7), default=None,
                   metavar="0..6", help="Development CV validation group (3 subjects); default original s07/s14/s21")
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--min-std", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=8.0)
    p.add_argument("--no-normalization", action="store_true",
                   help="Ablation: leave frozen embeddings on their original scales")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.topk not in range(1, 21):
        p.error("epochs and batch-size must be positive; topk must be between 1 and 20")
    if args.d_model % args.heads != 0:
        p.error("d-model must be divisible by heads")
    modes = expand_modes(args.modes)
    seed_everything(args.seed)

    cache = Path(args.cache)
    manifest = pd.read_csv(cache / "manifest.csv", dtype={"subject": str})
    train_man, val_man, validation_ids = split_development(manifest, args.reserve, args.val_group)
    print(
        f"TRIAL-LEVEL DEVELOPMENT ONLY: train subjects=18, validation={validation_ids}, "
        f"RESERVED TEST={args.reserve} (not loaded). "
        f"Windows train/val={len(train_man)}/{len(val_man)}, "
        f"trials train/val={len(group_trials(train_man))}/{len(group_trials(val_man))}."
    )
    print(f"Modes: {modes}; normalization={not args.no_normalization}", flush=True)
    if args.dry_run:
        return

    output_root = Path(args.out)
    if args.val_group is not None:
        output_root = output_root / f"dev_group_{args.val_group:02d}"
    output_root = output_root / f"seed{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    signature = {
        "cache": str(cache.resolve()),
        "reserve": args.reserve,
        "validation_group": args.val_group,
        "train_subjects": sorted(train_man["subject"].unique().tolist()),
        "validation_subjects": validation_ids,
        "seed": args.seed,
        "normalization_enabled": not args.no_normalization,
        "min_std": args.min_std,
        "clip": args.clip,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "topk": args.topk,
        "d_model": args.d_model,
        "heads": args.heads,
    }
    contract_path = output_root / "benchmark_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != signature:
            raise ValueError(
                f"Different experiment settings at {output_root}. "
                "Use a new --out directory to avoid mixing results."
            )
    else:
        contract_path.write_text(json.dumps(signature, indent=2))
    need_modes = []
    for mode in modes:
        path = output_root / mode / "validation_metrics.json"
        if args.skip_existing and path.is_file():
            print(f"SKIP completed {mode}: {path}", flush=True)
        else:
            need_modes.append(mode)
    if need_modes:
        print("Fitting feature normalizer on TRAIN subjects only...", flush=True)
        normalizer = FeatureStandardizer(min_std=args.min_std, clip=args.clip)
        normalizer.fit(cache, train_man, modalities=MODALITY_ORDER)
        if args.no_normalization:
            # Retain the fitted statistics in metadata for traceability but
            # do not apply them to the actual embeddings.
            applied_normalizer = None
        else:
            applied_normalizer = normalizer
        (output_root / "normalization_summary.json").write_text(
            json.dumps({
                "fitted_split": "train", "applied": not args.no_normalization,
                "per_modality": normalizer.summary(),
            }, indent=2)
        )
        print("Loading chronological trial embeddings; no encoder inference...", flush=True)
        train_ds = DEAPTrialDataset(cache, train_man, applied_normalizer)
        val_ds = DEAPTrialDataset(cache, val_man, applied_normalizer)
        print(
            f"Loaded {len(train_ds)} training trials and {len(val_ds)} validation trials; "
            f"token shapes={train_ds.shapes()}", flush=True
        )
        for mode in need_modes:
            result_dir = output_root / mode
            result_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*72}\nTRAINING MODE: {mode}\n{'='*72}", flush=True)
            if mode in RIDGE_MODES:
                metrics, pred = ridge_predict(mode, train_ds, val_ds)
                (result_dir / "validation_metrics.json").write_text(
                    json.dumps(metrics, indent=2)
                )
                pred.to_csv(result_dir / "validation_predictions.csv", index=False)
                print(f"{mode}: {json.dumps(metrics['subject_mean'])}", flush=True)
            else:
                # Models are fit anew for each mode on identical subjects and tokens.
                metrics = train_neural(
                    mode, train_ds, val_ds, applied_normalizer, args, result_dir
                )
            print(
                f"BEST VAL {mode}: subject mean={metrics['subject_mean']}; "
                f"pooled predicted labels={metrics['pooled']['predicted_class_counts']}",
                flush=True
            )
    leaderboard = []
    for mode in modes:
        met = json.loads(
            (output_root / mode / "validation_metrics.json").read_text()
        )
        leaderboard.append({
            "mode": mode,
            "subject_mean_bacc": met["subject_mean"]["balanced_accuracy"],
            "subject_mean_macro_f1": met["subject_mean"]["macro_f1"],
            "pooled_bacc": met["pooled"]["balanced_accuracy"],
            "pooled_macro_f1": met["pooled"]["macro_f1"],
            "validation_pred_0": met["pooled"]["predicted_class_counts"][0],
            "validation_pred_1": met["pooled"]["predicted_class_counts"][1],
        })
    final = pd.DataFrame(leaderboard).sort_values(
        ["subject_mean_macro_f1", "subject_mean_bacc"], ascending=False
    )
    final.to_csv(output_root / "leaderboard.csv", index=False)
    print("\nTRIAL-LEVEL DEVELOPMENT VALIDATION LEADERBOARD")
    print(final.round(4).to_string(index=False))
    print(f"\nSaved {output_root / 'leaderboard.csv'}")
    print(f"RESERVED {args.reserve} remains untouched. Validation metrics are "
          "used to choose checkpoints and are NOT final unbiased test scores.")


if __name__ == "__main__":
    main()
