"""Train/evaluate trial-level MIL with subject-independent LOSO."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from cmf.config import load_config
from cmf.datasets.deap_mil import DEAPTrialBagDataset, collate_trial_bags
from cmf.models.mil import DirectedFusionMIL
from cmf.utils.seed import seed_everything
from cmf.utils.io import ensure_dir


def _device_batch(batch, device):
    return ({k: v.to(device) for k, v in batch["modalities"].items()},
            {k: v.to(device) for k, v in batch["present"].items()},
            {k: v.to(device) for k, v in batch["quality_penalty"].items()},
            batch["mask"].to(device), batch["target"].to(device))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    actual, predicted, losses = [], [], []
    for batch in loader:
        mods, pres, qual, mask, labels = _device_batch(batch, device)
        logits, _ = model(mods, pres, qual, mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite MIL evaluation loss")
        actual.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(-1).cpu().tolist())
        losses.append((float(loss), len(labels)))
    return {"loss": sum(x*n for x, n in losses)/sum(n for _, n in losses),
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
            "n_trials": len(actual)}


def run_fold(config_path, manifest, test_subject, val_subject, seed):
    cfg = load_config(config_path)
    seed_everything(seed)
    cache = Path(cfg["dataset"]["cache_dir"])
    tc, mc = cfg["training"], cfg["model"]
    sc = cfg.get("loso", {}).get("subject_column", "subject")
    datasets = {}
    for split, subjects in (("train", [s for s in manifest[sc].unique() if s not in (test_subject, val_subject)]),
                            ("val", [val_subject]), ("test", [test_subject])):
        part = manifest[manifest[sc].isin(subjects)].copy()
        part["split"] = split
        datasets[split] = DEAPTrialBagDataset(
            cache, part, split, reliability_corruption=float(tc.get("reliability_corruption", 0))
            if split == "train" else 0,
            max_windows=cfg["dataset"].get("max_windows"))
    example = datasets["train"][0]
    shapes = {n: tuple(x.shape[1:]) for n, x in example["modalities"].items()}
    n_classes = int(cfg["dataset"].get("num_classes", 2))
    labels = np.array([bag[2] for bag in datasets["train"].bags])
    counts = np.bincount(labels, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("Empty training class: %s" % counts.tolist())
    weights = torch.tensor(len(labels)/(n_classes*counts), dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() and not tc.get("cpu", False) else "cpu")
    model = DirectedFusionMIL(shapes, n_classes, mc).to(device)
    loaders = {split: DataLoader(ds, batch_size=int(tc.get("batch_size", 4)),
                               shuffle=(split == "train"), num_workers=int(tc.get("workers", 0)),
                               collate_fn=collate_trial_bags)
               for split, ds in datasets.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc.get("lr", 1e-4)),
                                  weight_decay=float(tc.get("weight_decay", 1e-4)),
                                  eps=float(tc.get("adam_eps", 1e-7)))
    outdir = ensure_dir(Path(cfg.get("output_dir", "runs")) / cfg["dataset"]["name"] / "mil" /
                        ("%s_K%s_seed%s" % (mc.get("mode", "directed_topk"), mc.get("topk", 10), seed)) /
                        ("test_%s_val_%s" % (test_subject, val_subject)))
    history = []
    best = -1.0
    bad = 0
    patience = int(tc.get("patience", 30))
    for epoch in range(1, int(tc.get("epochs", 100))+1):
        start = time.time()
        model.train()
        total, grad_norms = [], []
        for batch_index, batch in enumerate(loaders["train"]):
            mods, pres, qual, mask, labels_batch = _device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(mods, pres, qual, mask)
            loss = torch.nn.functional.cross_entropy(logits, labels_batch, weight=weights.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MIL loss at epoch=%d batch=%d" % (epoch, batch_index))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(tc.get("grad_clip", 1.0)), error_if_nonfinite=True)
            optimizer.step()
            total.append(float(loss.detach()))
            grad_norms.append(float(norm))
        val = evaluate(model, loaders["val"], device)
        record = {"epoch": epoch, "train_loss": float(np.mean(total)),
                  "mean_grad_norm": float(np.mean(grad_norms)),
                  "max_grad_norm": float(np.max(grad_norms)),
                  "seconds": time.time()-start, **val}
        history.append(record)
        print(record, flush=True)
        if val["macro_f1"] > best+1e-6:
            best = val["macro_f1"]
            bad = 0
            torch.save({"model": model.state_dict(), "shapes": shapes, "config": cfg}, outdir/"best.pt")
        else:
            bad += 1
            if patience and bad >= patience:
                print("Early stopping at epoch", epoch)
                break
    (outdir/"history.json").write_text(json.dumps(history, indent=2))
    checkpoint = torch.load(outdir/"best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {"test_subject": test_subject, "val_subject": val_subject, "seed": seed,
              "best_val_macro_f1": best, "test": evaluate(model, loaders["test"], device),
              "outdir": str(outdir)}
    (outdir/"metrics.json").write_text(json.dumps(result, indent=2))
    return result


def run_loso_mil(config_path, subject=None, seed=None):
    cfg = load_config(config_path)
    manifest = pd.read_csv(Path(cfg["dataset"]["cache_dir"])/"manifest.csv")
    column = cfg.get("loso", {}).get("subject_column", "subject")
    if column not in manifest:
        raise ValueError("Missing subject column %s" % column)
    manifest[column] = manifest[column].astype(str)
    subjects = sorted(manifest[column].unique())
    if len(subjects) < 3:
        raise ValueError("LOSO requires at least three subjects")
    requested = [subject] if subject is not None else subjects
    if any(s not in subjects for s in requested):
        raise ValueError("Unknown subject: %s" % requested)
    seed = int(cfg.get("seed", 42) if seed is None else seed)
    results = []
    for test in requested:
        idx = subjects.index(test)
        val = subjects[(idx-1) % len(subjects)]
        print("MIL LOSO: test=%s val=%s" % (test, val), flush=True)
        results.append(run_fold(config_path, manifest, test, val, seed))
    if subject is None:
        summary = {"seed": seed, "folds": results}
        for metric in ("accuracy", "macro_f1"):
            values = [r["test"][metric] for r in results]
            summary[metric] = {"mean": float(np.mean(values)),
                               "std": float(np.std(values, ddof=1))}
        folder = ensure_dir(Path(cfg.get("output_dir", "runs"))/cfg["dataset"]["name"]/"mil")
        (folder/("loso_seed%d_summary.json" % seed)).write_text(json.dumps(summary, indent=2))
    return results
