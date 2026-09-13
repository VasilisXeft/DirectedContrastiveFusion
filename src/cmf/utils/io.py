from pathlib import Path
import json
import numpy as np


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_npz(path, modalities, target, meta=None):
    payload = {f"x__{k}": v for k, v in modalities.items()}
    payload["target"] = np.asarray(target)
    payload["meta_json"] = np.asarray(json.dumps(meta or {}))
    np.savez_compressed(path, **payload)


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    modalities = {k[3:]: z[k] for k in z.files if k.startswith("x__")}
    target = z["target"]
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
    return modalities, target, meta
