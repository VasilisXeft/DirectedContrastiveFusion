"""Preflight checks for the DEAP five-modality foundation benchmark."""
import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deap-dir", required=True)
    parser.add_argument("--video-manifest", required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    with open(args.video_manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    videos = {(r["subject"].lower(), int(r["trial"])): Path(r["video_path"]) for r in rows}
    subjects = args.subjects or sorted({s for s, _ in videos})
    total = 0
    for subject in subjects:
        path = Path(args.deap_dir) / f"{subject}.dat"
        if not path.is_file():
            print("MISSING PHYSIOLOGY", path)
            continue
        with path.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        signal = np.asarray(data["data"])
        labels = np.asarray(data["labels"])
        if signal.ndim != 3 or signal.shape[1] < 40 or labels.shape[0] != signal.shape[0]:
            raise ValueError(f"Unexpected DEAP data format: {subject}: {signal.shape} {labels.shape}")
        count = sum(1 for t in range(len(signal)) if (subject, t) in videos and videos[(subject, t)].is_file())
        total += count
        print(f"{subject}: signals={len(signal)}, matching videos={count}, "
              f"signal_shape={signal.shape}, label_shape={labels.shape}")
    print(f"Total matched subject/trials: {total}")
    print("Note: file matching does not verify physiological-video timestamp synchronization.")


if __name__ == "__main__":
    main()
