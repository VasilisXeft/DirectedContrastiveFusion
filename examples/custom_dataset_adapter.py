"""Minimal example: convert any aligned multimodal dataset to DCF cache format.

Replace `iter_my_samples()` with your own parser. The fusion/training code does not
need to know anything about the raw dataset once this cache has been created.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from cmf.utils.io import ensure_dir, save_npz

OUT = Path("data/processed/my_dataset")


def iter_my_samples():
    # Example only. Each modality is aligned/resampled to a fixed tensor shape.
    for i in range(100):
        yield {
            "sample_id": i,
            "subject": i % 10,
            "split": "train" if i < 70 else "val" if i < 85 else "test",
            "target": i % 4,
            "modalities": {
                "modality_a": np.random.randn(64, 8).astype(np.float32),
                "modality_b": np.random.randn(64, 6).astype(np.float32),
                "modality_c": np.random.randn(64, 10).astype(np.float32),
            },
            "missing_modalities": [],
        }


def main():
    samples_dir = ensure_dir(OUT / "samples")
    rows = []
    for item in iter_my_samples():
        rel = f"samples/{item['sample_id']:06d}.npz"
        meta = {
            "subject": item.get("subject"),
            "sample_id": item["sample_id"],
            "missing_modalities": item.get("missing_modalities", []),
        }
        save_npz(OUT / rel, item["modalities"], np.asarray(item["target"]), meta)
        rows.append({
            "sample_id": item["sample_id"],
            "subject": item.get("subject", -1),
            "label": item["target"] if np.asarray(item["target"]).ndim == 0 else -1,
            "split": item["split"],
            "file": rel,
        })
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)
    print(f"Wrote {len(rows)} samples to {OUT}")


if __name__ == "__main__":
    main()
