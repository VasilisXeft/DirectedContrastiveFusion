"""Audit *actual* checkpoint input contracts against our DEAP extraction.

Checks Hugging Face model configuration files, DEAP raw signal scales, and
cached feature values. This script does not infer a correct EEG amplitude unit,
change any cached sample, or fit transforms on held-out subjects.

Run:
  python scripts/audit_deap_foundation_inputs.py --deap-dir "D:/DEAP/data_preprocessed_python"
Optionally:
  python scripts/audit_deap_foundation_inputs.py --deap-dir "..." --video-manifest data/deap_video_manifest.csv
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def model_config(repository, filename):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repository, filename=filename)
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def summarize_array(name, array):
    x = np.asarray(array, dtype=np.float32)
    p = np.percentile(np.abs(x), [50, 95, 99])
    print(f"{name}: shape={x.shape}, finite={bool(np.isfinite(x).all())}, "
          f"mean={float(x.mean()):.6g}, std={float(x.std()):.6g}, "
          f"abs_p50/p95/p99={np.round(p, 7).tolist()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deap-dir", required=True)
    p.add_argument("--subject", default="s01")
    p.add_argument("--trial", type=int, default=0, help="Zero-based DEAP trial index")
    p.add_argument("--video-manifest", default=None)
    p.add_argument("--cache", default="data/features/deap_foundation_5m")
    args = p.parse_args()

    print("\n== EEGPT pretrained checkpoint ==")
    eeg = model_config("braindecode/eegpt-pretrained", "config.json")
    print("Released configuration:", {
        key: eeg.get(key) for key in [
            "n_chans", "sfreq", "n_times", "patch_size",
            "patch_stride", "embed_dim", "embed_num", "chan_proj_type"
        ]
    })
    from cmf.models.deap_eegpt import DEAP_CHANNELS
    from braindecode.models.eegpt import EEGPT_CHANNELS
    recognised = sorted(set(x.upper() for x in DEAP_CHANNELS) &
                        set(x.upper() for x in EEGPT_CHANNELS))
    unmapped = [x for x in DEAP_CHANNELS if x.upper() not in set(EEGPT_CHANNELS)]
    print(f"DEAP EEG channels={len(DEAP_CHANNELS)}, checkpoint-vocabulary overlap="
          f"{len(recognised)}, unmapped={unmapped}")
    print("LEGACY extractor: 128 -> 256 Hz, 4s=1024 samples; "
          "EEG amplitudes multiplied by 1e-6.")
    print("Original EEGPT pretraining preprocessing often converts to millivolt-scale "
          "values; confirm units and checkpoint convention before regenerating EEG.")
    if eeg.get("sfreq") != 256 or eeg.get("n_times") != 1024:
        print("WARNING: Our legacy preprocessing does not match hosted checkpoint "
              "sampling metadata. This may affect feature quality.")

    print("\n== MOMENT-1-large pretrained checkpoint ==")
    moment = model_config("AutonLab/MOMENT-1-large", "config.json")
    print("Config:", {
        "seq_len": moment.get("seq_len"),
        "patch_len": moment.get("patch_len"),
        "patch_stride_len": moment.get("patch_stride_len"),
        "embedding_dim": moment.get("t5_config", {}).get("d_model"),
    })
    print("Official momentfm embedding path expects float [B,C,512] with optional "
          "[B,512] mask, and applies internal RevIN.")
    print("LEGACY extractor: externally z-scores each 10s window, splits it into "
          "four 2.5s chunks, resamples each to 512 points. The external z-score "
          "is redundant with internal RevIN. Input 512 is satisfied.")

    print("\n== DINOv2-small pretrained image processor ==")
    vision = model_config("facebook/dinov2-small", "preprocessor_config.json")
    print("Processor:", {
        k: vision.get(k) for k in [
            "do_convert_rgb", "do_resize", "size", "do_center_crop", "crop_size",
            "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std"
        ]
    })
    print("Our extractor supplies RGB uint8 frames to AutoImageProcessor; "
          "the processor implements the checkpoint's spatial and color normalization.")
    print("CAUTION: DINOv2 is a generic vision encoder, not a facial-expression "
          "encoder; center-cropping entire video frames may exclude the face.")

    with (Path(args.deap_dir) / f"{args.subject}.dat").open("rb") as source:
        data = pickle.load(source, encoding="latin1")
    trial = np.asarray(data["data"][args.trial], dtype=np.float32)
    if trial.shape[0] != 40 or trial.shape[-1] != 8064:
        raise ValueError(f"Unexpected DEAP trial shape {trial.shape}")
    stim = trial[:, 3 * 128:]
    print(f"\n== Actual DEAP example ({args.subject}, trial {args.trial}) ==")
    summarize_array("EEG (raw stored values, 128Hz)", stim[:32])
    summarize_array("PPG (channel 38)", stim[38])
    summarize_array("EDA (channel 36)", stim[36])
    summarize_array("TEMP (channel 39)", stim[39])
    print("\n== Explicit legacy encoder-input tensor shapes ==")
    from scipy.signal import resample, resample_poly
    eeg_window = stim[:32, :10 * 128]
    eeg_256 = resample_poly(eeg_window * 1e-6, 2, 1, axis=-1)
    eeg_250 = resample_poly(eeg_window * 1e-3, 125, 64, axis=-1)
    print(f"EEGPT legacy 10s: {eeg_256.shape}, first 4s crop: "
          f"{eeg_256[:, :1024].shape} (32 channels @256Hz)")
    print(f"EEGPT checkpoint-aligned candidate 10s: {eeg_250.shape}, first 4s crop: "
          f"{eeg_250[:, :1000].shape} (32 channels @250Hz)")
    x = stim[38, :10 * 128].reshape(1, -1)
    x_legacy = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
    first_chunk = np.array_split(x_legacy, 4, axis=-1)[0]
    moment_tensor = resample(first_chunk, 512, axis=-1)[None].astype(np.float32)
    print(f"MOMENT legacy first 2.5s PPG input: {moment_tensor.shape} [B,C,L], "
          f"finite={bool(np.isfinite(moment_tensor).all())}")
    if moment_tensor.shape != (1, 1, 512):
        raise AssertionError("MOMENT input contract was violated")

    p95 = np.percentile(np.abs(stim[:32]), 95)
    print(f"EEG candidate p95 * 1e-6={p95 * 1e-6:.6g}; "
          f"/ 1000={p95 / 1000:.6g}. "
          "The two conventions differ by 1000x.")

    if args.video_manifest:
        with open(args.video_manifest, newline="", encoding="utf-8-sig") as source:
            rows = csv.DictReader(source)
            entry = next((r for r in rows if
                          r["subject"].lower().lstrip("s") == args.subject.lower().lstrip("s")
                          and int(r["trial"]) == args.trial), None)
        if entry is None:
            print("No matching video for selected trial.")
        else:
            import cv2
            video = Path(entry["video_path"])
            cap = cv2.VideoCapture(str(video))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, frame = cap.read()
            cap.release()
            print(f"Video: {video}, readable={ok}, fps={fps:.3f}, "
                  f"frames={total}, duration={total / fps if fps else float('nan'):.3f}s, "
                  f"raw_BGR_shape={frame.shape if ok else None}")
            if not ok or fps <= 0:
                raise RuntimeError(f"Unusable facial video: {video}")
            from PIL import Image
            import cv2
            from transformers import AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pixel = processor(images=[Image.fromarray(rgb)],
                              return_tensors="pt")["pixel_values"]
            print(f"DINOv2 actual preprocessed video frame shape={tuple(pixel.shape)}, "
                  f"min={float(pixel.min()):.4g}, max={float(pixel.max()):.4g}")
            if tuple(pixel.shape) != (1, 3, 224, 224):
                raise ValueError(f"Unexpected DINOv2 processor output shape: {pixel.shape}")

    cache = Path(args.cache)
    matching = sorted(cache.glob(f"{args.subject}_trial{args.trial:02d}_start*.npz"))
    if matching:
        print("\n== Existing cached frozen embeddings ==")
        with np.load(matching[0], allow_pickle=False) as z:
            for name in ("eeg", "ppg", "eda", "temp", "face"):
                summarize_array(name, z["x__" + name])
        print("These are *output* scales; correct their cross-modal imbalance "
              "using a training-only feature normalizer, NOT by per-sample z-score.")
    else:
        print(f"No existing cached example for {args.subject}/trial{args.trial}.")

    print("\nAUDIT COMPLETE. No extraction or model training was performed.")


if __name__ == "__main__":
    main()
