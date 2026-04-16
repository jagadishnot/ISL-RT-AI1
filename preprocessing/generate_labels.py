import os
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

VIDEO_ROOT = "data/videos/Videos_Sentence_Level"
LANDMARK_DIR = "data/landmarks"
OUTPUT_FILE = "data/labels.csv"

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def clean_sentence(sentence):
    """
    Normalize sentence text.
    """
    sentence = sentence.lower()
    sentence = sentence.strip()
    sentence = sentence.replace("_", " ")
    sentence = " ".join(sentence.split())
    return sentence


# ---------------------------------------------------------
# LOAD LANDMARK FILES
# ---------------------------------------------------------

landmark_files = sorted([
    f for f in os.listdir(LANDMARK_DIR)
    if f.endswith(".npy")
])

print("Total landmark files:", len(landmark_files))


# ---------------------------------------------------------
# COLLECT VIDEO DATA
# ---------------------------------------------------------

video_data = []

for sentence_folder in sorted(os.listdir(VIDEO_ROOT)):

    folder_path = os.path.join(VIDEO_ROOT, sentence_folder)

    if not os.path.isdir(folder_path):
        continue

    sentence = clean_sentence(sentence_folder)

    for video_file in sorted(os.listdir(folder_path)):

        if video_file.lower().endswith(".mp4"):

            video_path = os.path.join(folder_path, video_file)

            video_data.append({
                "sentence": sentence,
                "video_file": video_file,
                "video_path": video_path
            })

print("Total videos found:", len(video_data))


# ---------------------------------------------------------
# MATCH LANDMARKS WITH VIDEOS
# ---------------------------------------------------------

rows = []

limit = min(len(video_data), len(landmark_files))

for i in range(limit):

    rows.append({
        "video": landmark_files[i],
        "text": video_data[i]["sentence"]
    })


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame(rows)

# Remove duplicates if any
df = df.drop_duplicates()

# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

os.makedirs("data", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

# ---------------------------------------------------------
# DATASET STATISTICS
# ---------------------------------------------------------

print("\nDataset Summary")
print("---------------------------")

print("Total samples:", len(df))

print("Unique sentences:", df["text"].nunique())

print("\nTop sentence counts:")

print(df["text"].value_counts().head(10))

print("\nLabels saved to:", OUTPUT_FILE)