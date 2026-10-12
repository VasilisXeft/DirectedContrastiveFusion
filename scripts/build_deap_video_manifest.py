"""Inspect local DEAP videos and generate a subject/trial manifest.

This script never guesses trial numbering or synchronization offsets.
"""
import argparse
import csv
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out", default="data/deap_video_manifest.csv")
    p.add_argument("--trial-base", type=int, choices=[0, 1], default=None,
                   help="Explicitly state whether video trial numbers start at 0 or 1")
    a = p.parse_args()
    root = Path(a.video_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    extensions = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}
    videos = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    if not videos:
        raise RuntimeError(f"No video files found in {root}")
    print(f"Found {len(videos)} video files")
    for video in videos[:25]:
        print("EXAMPLE", video.relative_to(root))
    if a.trial_base is None:
        print("Inspection only. Re-run with --trial-base 0 or 1 after verifying the names.")
        return
    rows = []
    unmatched = []
    duplicate = []
    seen = set()
    for video in videos:
        # Match explicit subject/trial tokens from the filename or parent directories.
        name = " ".join(video.relative_to(root).parts)
        subject = re.search(r"(?<![a-z0-9])(?:subject|sub|s)[_ -]*0*([1-9]|[12][0-9]|3[012])(?![0-9])", name, re.I)
        trial = re.search(r"(?<![a-z0-9])(?:trial|video|v)[_ -]*0*(\d+)(?![0-9])", name, re.I)
        if not subject or not trial:
            unmatched.append(str(video))
            continue
        sid = f"s{int(subject.group(1)):02d}"
        tid = int(trial.group(1)) - a.trial_base
        key = (sid, tid)
        if tid < 0 or tid >= 40 or key in seen:
            duplicate.append((key, str(video)))
            continue
        seen.add(key)
        rows.append({"subject": sid, "trial": tid, "video_path": str(video.resolve())})
    print(f"Matched {len(rows)}; unmatched {len(unmatched)}; invalid/duplicate {len(duplicate)}")
    for value in unmatched[:20]:
        print("UNMATCHED", value)
    for value in duplicate[:20]:
        print("INVALID_OR_DUPLICATE", value)
    if unmatched or duplicate:
        raise RuntimeError("Ambiguous video filenames: manifest not written. Inspect examples and adapt matching.")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "trial", "video_path"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["subject"], r["trial"])))
    print("WROTE", out)


if __name__ == "__main__":
    main()
