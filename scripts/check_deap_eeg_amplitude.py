"""Inspect the physical scale of DEAP EEG before using pretrained EEGPT.

Does not rewrite features. Does not inspect emotion labels.

Example:
python scripts/check_deap_eeg_amplitude.py --deap-dir "C:/path/to/data_preprocessed_python"
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deap-dir", required=True)
    p.add_argument("--subjects", nargs="+", default=["s01", "s02", "s03"])
    p.add_argument("--trials", type=int, default=5)
    args = p.parse_args()

    for subject in args.subjects:
        source = Path(args.deap_dir) / f"{subject}.dat"
        with source.open("rb") as f:
            raw = pickle.load(f, encoding="latin1")
        x = np.asarray(raw["data"][:args.trials, :32, 3 * 128:], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite raw EEG: {subject}")
        magnitude = np.percentile(np.abs(x), [50, 95, 99, 99.9])
        channel_sd = x.std(axis=-1)
        print(f"\n{subject}: {x.shape[0]} trials, {x.shape[1]} EEG channels, "
              f"{x.shape[2]} samples")
        print(f"raw absolute amplitudes [p50,p95,p99,p99.9]: {magnitude}")
        print(f"raw channel standard deviation: median={np.median(channel_sd):.6g}, "
              f"p95={np.percentile(channel_sd,95):.6g}")
        print("Candidate EEGPT input p95 amplitudes (no assumption about units):")
        for name, scale in [
            ("raw * 1e-6 (CURRENT)", 1e-6),
            ("raw / 1000", 1e-3),
            ("raw unscaled", 1.0),
        ]:
            print(f"  {name}: {magnitude[1]*scale:.8g}")

    print("\nInterpretation requires confirming the stored DEAP EEG unit AND "
          "the preprocessing convention used to pretrain the selected EEGPT checkpoint.")
    print("Do not choose a scale merely because it produces larger feature variance.")


if __name__ == "__main__":
    main()
