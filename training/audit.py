import numpy as np
import pandas as pd
import os

base_dir     = "."
landmark_dir = os.path.join(base_dir, "data", "landmarks")
labels_path  = os.path.join(base_dir, "data", "labels.csv")

df = pd.read_csv(labels_path)

tiny   = 0   # std < 0.001  (over-normalized / near zero)
heavy_zeros = 0  # zeros > 50%
normal = 0   # looks correct
total  = len(df)

tiny_examples   = []
zeros_examples  = []

for i, row in df.iterrows():
    path = os.path.join(landmark_dir, row["video"])
    if not os.path.exists(path):
        continue

    data      = np.load(path).astype(np.float32)
    std       = data.std()
    zero_pct  = (data == 0).mean()

    if std < 0.001:
        tiny += 1
        if len(tiny_examples) < 3:
            tiny_examples.append((row["video"], std, zero_pct))

    elif zero_pct > 0.5:
        heavy_zeros += 1
        if len(zeros_examples) < 3:
            zeros_examples.append((row["video"], std, zero_pct))

    else:
        normal += 1

print("=== CONSISTENCY AUDIT ===")
print(f"Total files      : {total}")
print(f"Normal           : {normal}  ({normal/total*100:.1f}%)")
print(f"Tiny std (<0.001): {tiny}  ({tiny/total*100:.1f}%)  ← bad")
print(f"Heavy zeros(>50%): {heavy_zeros}  ({heavy_zeros/total*100:.1f}%)  ← bad")

print("\nTiny std examples:")
for f, s, z in tiny_examples:
    print(f"  {f}  std={s:.6f}  zeros={z*100:.1f}%")

print("\nHeavy zeros examples:")
for f, s, z in zeros_examples:
    print(f"  {f}  std={s:.4f}  zeros={z*100:.1f}%")