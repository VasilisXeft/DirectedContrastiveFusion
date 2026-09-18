import os
import pickle
import cv2
import numpy as np

DEAP_DATA_PATH = (
    r"C:\Users\vxefteris\Desktop\D\MindSpaces\DEAP Dataset"
    r"\data_preprocessed_python\data_preprocessed_python"
)

DEAP_VIDEO_PATH = (
    r"C:\Users\vxefteris\Desktop\D\MindSpaces\DEAP Dataset\face_video"
)


def main():
    # ---------------------------------------------------------
    # 1. Check physiological/EEG files
    # ---------------------------------------------------------
    dat_files = sorted(
        f for f in os.listdir(DEAP_DATA_PATH)
        if f.endswith(".dat")
    )

    print("=" * 70)
    print("DEAP DATA")
    print("=" * 70)
    print(f"Subjects found: {len(dat_files)}")
    print(dat_files)

    # ---------------------------------------------------------
    # 2. Inspect subject 01
    # ---------------------------------------------------------
    path = os.path.join(DEAP_DATA_PATH, "s01.dat")

    with open(path, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    print("\n" + "=" * 70)
    print("s01.dat")
    print("=" * 70)

    print("Keys:", subject.keys())

    data = np.asarray(subject["data"])
    labels = np.asarray(subject["labels"])

    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    print("Data dtype:", data.dtype)
    print("Labels dtype:", labels.dtype)

    print("\nFirst trial labels:")
    print(labels[0])

    print("\nLabel ranges:")
    for i, name in enumerate(
        ["Valence", "Arousal", "Dominance", "Liking"]
    ):
        print(
            f"{name:10s}: "
            f"min={labels[:, i].min():.3f}, "
            f"max={labels[:, i].max():.3f}, "
            f"mean={labels[:, i].mean():.3f}"
        )

    # ---------------------------------------------------------
    # 3. Channel sanity
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("CHANNEL / SIGNAL CHECK")
    print("=" * 70)

    print("Trials:", data.shape[0])
    print("Channels:", data.shape[1])
    print("Samples/trial:", data.shape[2])

    print("\nFirst trial channel statistics:")
    for ch in range(data.shape[1]):
        x = data[0, ch]

        print(
            f"ch {ch:02d}: "
            f"mean={x.mean(): .5f}, "
            f"std={x.std(): .5f}, "
            f"min={x.min(): .5f}, "
            f"max={x.max(): .5f}"
        )

    # ---------------------------------------------------------
    # 4. Video inventory
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("VIDEO CHECK")
    print("=" * 70)

    subjects_with_video = sorted(
        d for d in os.listdir(DEAP_VIDEO_PATH)
        if os.path.isdir(os.path.join(DEAP_VIDEO_PATH, d))
    )

    print(f"Video subject folders: {len(subjects_with_video)}")

    total_videos = 0
    problems = []

    for subject_id in subjects_with_video:

        folder = os.path.join(DEAP_VIDEO_PATH, subject_id)

        videos = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".avi", ".mp4"))
        )

        total_videos += len(videos)

        if len(videos) != 40:
            problems.append((subject_id, len(videos)))

    print(f"Total videos: {total_videos}")

    if problems:
        print("\nSubjects without exactly 40 videos:")
        for p in problems:
            print(p)
    else:
        print("All video subjects contain exactly 40 trials.")

    # ---------------------------------------------------------
    # 5. Inspect s01 trial01 video
    # ---------------------------------------------------------
    video_path = os.path.join(
        DEAP_VIDEO_PATH,
        "s01",
        "s01_trial01.avi"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("\nERROR: Could not open:", video_path)
    else:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frames / fps if fps > 0 else float("nan")

        print("\nExample video:")
        print("Path:", video_path)
        print("Frames:", frames)
        print("FPS:", fps)
        print("Resolution:", f"{width}x{height}")
        print("Duration:", round(duration, 2), "s")

        ret, frame = cap.read()

        if ret:
            print("First frame shape:", frame.shape)
        else:
            print("WARNING: first frame could not be read.")

        cap.release()

    # ---------------------------------------------------------
    # 6. Exact trial mapping
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRIAL MAPPING")
    print("=" * 70)

    missing = []

    for sid in range(1, 33):

        subject_id = f"s{sid:02d}"

        dat_path = os.path.join(
            DEAP_DATA_PATH,
            f"{subject_id}.dat"
        )

        if not os.path.exists(dat_path):
            missing.append(f"Missing DAT: {subject_id}")
            continue

        for trial in range(1, 41):

            video = os.path.join(
                DEAP_VIDEO_PATH,
                subject_id,
                f"{subject_id}_trial{trial:02d}.avi"
            )

            if not os.path.exists(video):
                missing.append(
                    f"Missing video: {subject_id} trial {trial}"
                )

    if missing:
        print("Mapping problems:")
        for x in missing[:100]:
            print(x)
    else:
        print("All 32 x 40 DAT/video trial mappings exist.")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()