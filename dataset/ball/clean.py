import pandas as pd
import numpy as np
from os import listdir, remove, path

df = pd.read_csv("rows.csv")
print(df[(df.release_mp4_frame >= 165) & (df.release_mp4_frame <= 195)].release_mp4_frame.describe())

plays = []
frames = []

for filename in listdir("images/train"):
    splits = filename.split(".")
    play = splits[0][len("play_"):]
    frame = np.int64(int(splits[1][len("frame_"):]))
    plays.append(play)
    frames.append(frame)

columns = np.column_stack((np.array(plays), np.array(frames, dtype=np.int64)))
columns = pd.DataFrame(data=columns, columns=["play_id", "mp4_frame"]).astype({"play_id": str, "mp4_frame": np.int64})

columns_all = columns.merge(df[["play_id", "mp4_frame"]], on=["play_id", "mp4_frame"], how='left', indicator=True)
columns_only = columns_all[columns_all["_merge"] == "left_only"]

bad_release_mp4_frame = df[(df.release_mp4_frame < 165) | (df.release_mp4_frame > 195)][["play_id", "mp4_frame"]]

to_delete = pd.concat((columns_only, bad_release_mp4_frame), ignore_index=True)

for row in to_delete.iloc:
    play_id = row.play_id
    mp4_frame = row.mp4_frame
    filename = f"play_{play_id}.frame_{mp4_frame:03}"
    if path.exists(f"images/train/{filename}.png"):
        print(f"Deleting {filename}...")
        remove(f"images/train/{filename}.png")
    if path.exists(f"labels/train/{filename}.txt"):
        remove(f"labels/train/{filename}.txt")
