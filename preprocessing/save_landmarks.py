import cv2
import os
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import sys

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

VIDEO_ROOT = "data/videos/Videos_Sentence_Level"
OUTPUT_DIR = "data/landmarks"
LABELS_CSV = "data/labels.csv"
PROGRESS_FILE = "data/landmarks_progress.txt"  # tracks completed videos

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 478 face (refine=True) + 21 left + 21 right + 33 pose = 1659
EXPECTED_FEATURES = 1659

# ---------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------

def extract_landmarks(results):

    landmarks = []

    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (478 * 3))

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (21 * 3))

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (21 * 3))

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (33 * 3))

    return landmarks


# ---------------------------------------------------------
# Process single video (called in subprocess)
# ---------------------------------------------------------

def process_video(args):

    idx, video_path, label = args

    save_name = f"video_{idx}.npy"
    save_path = os.path.join(OUTPUT_DIR, save_name)

    # RESUME: skip if already successfully saved
    if os.path.exists(save_path):
        data = np.load(save_path)
        if data.std() > 0.01:
            return (idx, save_name, label, "skipped_done")

    mp_holistic = mp.solutions.holistic

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            refine_face_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as holistic:

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return (idx, save_name, label, "failed")

            sequence   = []
            last_valid = None

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                try:
                    frame = cv2.resize(frame, (640, 480))
                    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True

                    landmarks = extract_landmarks(results)
                    arr = np.array(landmarks, dtype=np.float32)

                    if np.abs(arr).sum() < 1e-6:
                        if last_valid is not None:
                            arr = last_valid.copy()
                    else:
                        last_valid = arr.copy()

                    sequence.append(arr)

                except:
                    if last_valid is not None:
                        sequence.append(last_valid.copy())
                    else:
                        sequence.append(np.zeros(EXPECTED_FEATURES, dtype=np.float32))

            cap.release()

            if len(sequence) == 0:
                return (idx, save_name, label, "failed")

            sequence = np.stack(sequence, axis=0)

            if sequence.std() < 0.01:
                return (idx, save_name, label, "low_signal")

            np.save(save_path, sequence)
            return (idx, save_name, label, "saved")

    except Exception as e:
        return (idx, save_name, label, f"error: {e}")


# ---------------------------------------------------------
# Collect videos
# ---------------------------------------------------------

def collect_videos():

    entries = []

    for label_folder in sorted(os.listdir(VIDEO_ROOT)):

        folder_path = os.path.join(VIDEO_ROOT, label_folder)

        if not os.path.isdir(folder_path):
            continue

        label = label_folder.lower().strip()

        for file in sorted(os.listdir(folder_path)):
            if file.lower().endswith((".mp4", ".avi", ".mov", ".MP4")):
                entries.append((os.path.join(folder_path, file), label))

    return entries


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    entries = collect_videos()
    total   = len(entries)
    print(f"Found {total} videos")
    print(f"Feature dim: {EXPECTED_FEATURES}")

    # build args list with index
    args_list = [(idx, vp, lb) for idx, (vp, lb) in enumerate(entries)]

    # count already done
    already_done = sum(
        1 for idx, vp, lb in args_list
        if os.path.exists(os.path.join(OUTPUT_DIR, f"video_{idx}.npy")) and
           np.load(os.path.join(OUTPUT_DIR, f"video_{idx}.npy")).std() > 0.01
    )
    print(f"Already done: {already_done} / {total}  (will resume from here)\n")

    # use 8 workers
    workers = min(6, cpu_count())
    print(f"Using {workers} CPU cores\n")

    rows    = {}
    saved   = 0
    skipped = 0
    failed  = 0

    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap(process_video, args_list),
            total=total
        ):
            idx, save_name, label, status = result

            if status in ("saved", "skipped_done"):
                rows[idx] = {"video": save_name, "text": label}
                if status == "saved":
                    saved += 1
                else:
                    skipped += 1
            else:
                failed += 1

    # write labels.csv sorted by index
    df = pd.DataFrame([rows[k] for k in sorted(rows.keys())])
    df.to_csv(LABELS_CSV, index=False)

    print(f"\nExtraction complete")
    print(f"Saved   : {saved}")
    print(f"Resumed : {skipped}")
    print(f"Failed  : {failed}")
    print(f"labels.csv written with {len(df)} rows")

    # sanity check on first saved file
    if rows:
        first = rows[min(rows.keys())]
        sample = np.load(os.path.join(OUTPUT_DIR, first["video"]))
        print(f"\nSample check — {first['video']}:")
        print(f"  shape : {sample.shape}")
        print(f"  std   : {sample.std():.4f}  ← should be > 0.01")


if __name__ == "__main__":
    main()