from pathlib import Path
import numpy as np
import pandas as pd
from cmf.utils.io import ensure_dir, save_npz


def preprocess_synthetic(cfg):
    dcfg = cfg["dataset"]
    out = ensure_dir(dcfg["cache_dir"])
    samples = ensure_dir(out / "samples")
    rng = np.random.default_rng(cfg.get("seed", 42))
    n = int(dcfg.get("n_samples", 240))
    n_classes = int(dcfg.get("num_classes", 4))
    rows = []
    for i in range(n):
        y = i % n_classes
        latent = rng.normal(loc=y * 0.5, scale=1.0, size=(64, 8)).astype(np.float32)
        mods = {
            "sensor_a": latent + 0.2 * rng.normal(size=latent.shape),
            "sensor_b": latent[:, :6] + 0.25 * rng.normal(size=(64, 6)),
            "sensor_c": np.concatenate([latent[:, :4], rng.normal(size=(64, 2))], axis=1),
            "sensor_d": latent[:, 2:8] + 0.35 * rng.normal(size=(64, 6)),
        }
        mods = {k: v.astype(np.float32) for k, v in mods.items()}
        split = "train" if i < int(.7*n) else "val" if i < int(.85*n) else "test"
        rel = f"samples/{i:05d}.npz"
        save_npz(out / rel, mods, np.array(y, dtype=np.int64), {"subject": i % 20, "id": i})
        rows.append({"sample_id": i, "subject": i % 20, "label": y, "split": split, "file": rel})
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    return out
