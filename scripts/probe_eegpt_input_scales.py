"""Compare EEGPT feature sensitivity to raw-DEAP amplitude conventions.

Unsupservised *development-only* diagnostic: does not read valence labels or
reconfigure an existing cache. Larger feature variance is NOT evidence of
better predictive generalization. Use only subjects excluded from final test.

Example:
 python scripts/probe_eegpt_input_scales.py --deap-dir "D:/DEAP/data_preprocessed_python" --subjects s01 s02
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from cmf.models.deap_eegpt import EEGPTFeatures


def describe(features):
    data = np.stack(features).astype(np.float32)
    mu = data.mean(axis=0)
    return {
        "vectors": len(data),
        "feature_dim": int(data.shape[-1]),
        "between_window_std_mean": float(data.std(axis=0).mean()),
        "between_window_std_median": float(np.median(data.std(axis=0))),
        "mean_vector_norm": float(np.linalg.norm(data, axis=1).mean()),
        "near_constant_features": int((data.std(axis=0) < 1e-6).sum()),
        "pooled_feature_mean_abs": float(np.abs(mu).mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deap-dir", required=True)
    p.add_argument("--subjects", nargs="+", default=["s01", "s02"])
    p.add_argument("--trials-per-subject", type=int, default=2)
    p.add_argument("--windows-per-trial", type=int, default=3)
    p.add_argument("--target-sfreq", type=int, default=250, choices=[250, 256])
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if args.trials_per_subject < 1 or args.windows_per_trial < 1:
        raise ValueError("trials-per-subject and windows-per-trial must be positive")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EEGPTFeatures(device=device, target_sfreq=args.target_sfreq, input_scale=1e-6)
    inputs = []
    for subject in args.subjects:
        with (Path(args.deap_dir) / f"{subject}.dat").open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        for trial in np.asarray(data["data"])[:args.trials_per_subject]:
            signal = np.asarray(trial[:32, 3 * 128:], dtype=np.float32)
            for start in np.linspace(0, signal.shape[-1] - 10 * 128,
                                     args.windows_per_trial).round().astype(int):
                inputs.append(signal[:, start:start + 10 * 128])

    result = {}
    for name, scale in [("legacy_volt_scale_1e-6", 1e-6),
                        ("candidate_millivolt_scale_1e-3", 1e-3)]:
        encoder.input_scale = scale
        vectors = []
        for i, signal in enumerate(inputs, 1):
            embeddings = encoder(signal)
            if embeddings.shape != (encoder.tokens, 512):
                raise ValueError(f"Unexpected EEGPT features: {embeddings.shape}")
            vectors.append(embeddings.mean(axis=0))
            if i % 10 == 0:
                print(f"{name}: processed {i}/{len(inputs)} windows", flush=True)
        result[name] = np.stack(vectors)
        print(f"\n{name}: {describe(vectors)}", flush=True)

    a, b = result.values()
    diff = np.linalg.norm(a - b, axis=1)
    print("\nScale sensitivity (same EEG samples, fixed pretrained weights):")
    print(f"Average embedding L2 difference: {float(diff.mean()):.6g}")
    print(f"Average difference / legacy norm: "
          f"{float((diff / np.maximum(np.linalg.norm(a, axis=1), 1e-8)).mean()):.6g}")
    print("NO labels used, no cache modified. These diagnostics assess representation "
          "sensitivity, not which scale predicts emotions better. Verify source units "
          "and use subject-disjoint development performance to choose preprocessing.")


if __name__ == "__main__":
    main()
