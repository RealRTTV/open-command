import pandas as pd
import numpy as np
from os import listdir, remove, path

df = pd.read_csv("rows_train.csv")

to_delete = df[df.baseball_center_x < 0]

print(to_delete)
input("Press Enter to continue...")

for row in to_delete.iloc:
    play_id = row.play_id
    mp4_frame = row.mp4_frame
    filename = f"play_{play_id}.frame_{mp4_frame:03}"
    if path.exists(f"images/train/{filename}.png"):
        print(f"Deleting {filename}...")
        remove(f"images/train/{filename}.png")
    if path.exists(f"labels/train/{filename}.txt"):
        remove(f"labels/train/{filename}.txt")
