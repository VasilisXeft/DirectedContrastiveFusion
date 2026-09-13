from pathlib import Path
import pandas as pd
from cmf.utils.io import load_npz


def preprocess_generic(cfg):
    """Validate an already prepared dataset cache.

    This is the dataset-agnostic entry point. A cache must contain:
      cache_dir/manifest.csv
      cache_dir/<sample files>.npz

    manifest.csv requires columns: file, split. For classification, a label column
    is strongly recommended because it is used to infer num_classes when the config
    does not specify it.

    Every .npz sample is created with cmf.utils.io.save_npz and contains arbitrary
    named modalities. Numeric/time-series modalities should be [T, F]; image/video
    modalities should be [T, H, W, C]. All samples must expose the same modality
    names and tensor shapes after preprocessing. Missing modalities are represented
    with zero-valued arrays plus meta['missing_modalities'].
    """
    cache = Path(cfg["dataset"]["cache_dir"])
    manifest_path = cache / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Build the generic cache first; see "
            "examples/custom_dataset_adapter.py and docs/CUSTOM_DATASETS.md"
        )
    manifest = pd.read_csv(manifest_path)
    required = {"file", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest.csv missing required columns: {sorted(missing)}")
    if len(manifest) == 0:
        raise ValueError("manifest.csv contains no samples")

    first = cache / str(manifest.iloc[0]["file"])
    if not first.exists():
        raise FileNotFoundError(first)
    modalities, _, _ = load_npz(first)
    if len(modalities) < 2:
        raise ValueError("Directed multimodal fusion requires at least two modalities")
    return cache
