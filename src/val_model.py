import time

from ultralytics import YOLO
import pandas as pd
import numpy as np
import os

OBJECT = "glove"

start = time.time()
model = YOLO("../models/glove.engine")
results = model.predict("../dataset/glove/images/val", imgsz=704, max_det=1, conf=0.01, batch=16, half=True)
print(f"took {time.time() - start}s")
raw = pd.read_csv('../dataset/glove/rows_val.csv')
model_data = pd.DataFrame([{
    f"model_{OBJECT}_center_x": np.append(r.boxes.xywh[:, 0].cpu().numpy(), np.nan)[0],
    f"model_{OBJECT}_center_y": np.append(r.boxes.xywh[:, 1].cpu().numpy(), np.nan)[0],
    "play_id": os.path.basename(r.path).split(".")[0][len("play_"):],
    "mp4_frame": int(os.path.basename(r.path).split(".")[1][len("frame_"):]),
} for r in results])

df = pd.merge(raw, model_data, on=["play_id", "mp4_frame"], how='left')

tp = df[~np.isnan(df[f"{OBJECT}_center_x"]) & ~np.isnan(df[f"model_{OBJECT}_center_x"])]

dist = np.sqrt((tp[f"model_{OBJECT}_center_x"].to_numpy() - tp[f"{OBJECT}_center_x"].to_numpy()) ** 2 + (tp[f"model_{OBJECT}_center_y"].to_numpy() - tp[f"{OBJECT}_center_y"].to_numpy()) ** 2)

tn = df[np.isnan(df[f"{OBJECT}_center_x"]) & np.isnan(df[f"model_{OBJECT}_center_x"])]
fp = df[~np.isnan(df[f"{OBJECT}_center_x"]) & np.isnan(df[f"model_{OBJECT}_center_x"])]
fn = df[np.isnan(df[f"{OBJECT}_center_x"]) & ~np.isnan(df[f"model_{OBJECT}_center_x"])]

print(f"tp: {len(tp)}, tn: {len(tn)}, fp: {len(fp)}, fn: {len(fn)}")
print(f"fp:\n{fp[["play_id", "mp4_frame", "game_pk", "frame_idx"]]}")
print(f"fn:\n{fn[["play_id", "mp4_frame", "game_pk", "frame_idx"]]}")
print(f"dist: median: {np.median(dist)}, p90: {np.percentile(dist, 90)}")
