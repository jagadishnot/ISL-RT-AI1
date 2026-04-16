import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset

MAX_FRAMES = 120
FEATURE_DIM = 1659


class CSLTDataset(Dataset):

    def __init__(self, augment=False):

        base_dir = os.path.dirname(os.path.dirname(__file__))

        labels_path = os.path.join(base_dir, "data", "labels.csv")
        self.landmark_dir = os.path.join(base_dir, "data", "landmarks")

        self.df = pd.read_csv(labels_path)

        # drop rows where landmark file is missing
        self.df = self.df[
            self.df["video"].apply(
                lambda f: os.path.exists(os.path.join(self.landmark_dir, f))
            )
        ].reset_index(drop=True)

        self.augment = augment

    def __len__(self):
        return len(self.df)

    def pad_sequence(self, seq):

        seq_len = seq.shape[0]

        if seq_len >= MAX_FRAMES:
            return seq[:MAX_FRAMES], MAX_FRAMES

        pad_len = MAX_FRAMES - seq_len
        padding = np.zeros((pad_len, FEATURE_DIM), dtype=np.float32)

        # FIX: return actual length alongside padded sequence
        return np.concatenate((seq, padding), axis=0), seq_len

    def normalize(self, seq, actual_len):

        # only normalize real frames, not padding
        valid = seq[:actual_len]

        mean = valid.mean(axis=0)
        std  = valid.std(axis=0)
        std[std < 1e-6] = 1.0

        seq[:actual_len] = (valid - mean) / std

        return seq

    def temporal_augment(self, seq):

        if len(seq) < 20:
            return seq

        scale   = np.random.uniform(0.8, 1.2)
        new_len = max(int(len(seq) * scale), 1)
        indices = np.linspace(0, len(seq)-1, new_len).astype(int)

        return seq[indices]

    def spatial_augment(self, seq):
        noise = np.random.normal(0, 0.01, seq.shape).astype(np.float32)
        return seq + noise

    def motion_augment(self, seq):

        if len(seq) < 3:
            return seq

        velocity = np.diff(seq, axis=0)
        velocity = np.vstack((velocity, velocity[-1]))

        return seq + 0.2 * velocity

    def __getitem__(self, idx):

        row       = self.df.iloc[idx]
        video_file = row["video"]
        text      = str(row["text"]).lower().strip()

        path = os.path.join(self.landmark_dir, video_file)

        try:
            data = np.load(path).astype(np.float32)

            if data.ndim != 2 or data.shape[1] != FEATURE_DIM:
                data = np.zeros((MAX_FRAMES, FEATURE_DIM), dtype=np.float32)

        except Exception:
            data = np.zeros((MAX_FRAMES, FEATURE_DIM), dtype=np.float32)

        if self.augment:
            data = self.temporal_augment(data)

            if np.random.rand() < 0.5:
                data = self.spatial_augment(data)

            if np.random.rand() < 0.3:
                data = self.motion_augment(data)

        # FIX: get actual length before padding
        data, actual_len = self.pad_sequence(data)

        data = self.normalize(data, actual_len)

        data = torch.from_numpy(data)

        # FIX: return actual_len so train.py can pass correct input_lengths to CTC
        return data, text, actual_len   