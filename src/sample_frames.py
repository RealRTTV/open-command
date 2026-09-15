import datetime
import random
from tqdm import tqdm
from itertools import chain
from typing import Optional

import pandas as pd
import numpy as np
import cv2
import os
import platform

SS_CACHE_DIR: str = "/Users/riley/Library/Caches/statcast-subsidiary" if platform.system() == "Darwin" else "/home/riley/.cache/statcast-subsidiary"
BASEBALL_WIDTH: int = 20
BASEBALL_HEIGHT: int = 20

sz_df = pd.read_csv("../data/2026/raw/strikezone_tracking.csv.gz", compression="gzip")

class StrikeZoneEntry:
    x_left: np.float64
    x_right: np.float64
    y_top: np.float64
    y_bottom: np.float64

    def __init__(self):
        self.x_left = np.nan
        self.x_right = np.nan
        self.y_top = np.nan
        self.y_bottom = np.nan

# todo: remove duplicates by pruning duplicate rows later
# game_pk,play_id,mp4_path,mp4_frame,frame_idx,release_mp4_frame,glove_center_x,glove_center_y,baseball_center_x,baseball_center_y,sz_x_left,sz_y_top,sz_x_right,sz_y_bottom
def get_row_for_sample(date: str, hardcoded_play_id: Optional[str] = None) -> Optional[str]:
    date: datetime.date = datetime.datetime.strptime(date, "%Y-%m-%d")
    play_ids_for_date_path: str = os.path.join(SS_CACHE_DIR, f"statcast-play-ids/{date.year}/{date.strftime("%Y-%m-%d")}.csv")
    if not os.path.exists(play_ids_for_date_path) or os.path.getsize(play_ids_for_date_path) == 0:
        return None
    df: pd.DataFrame = pd.read_csv(play_ids_for_date_path)
    df.dropna(subset=["playId"], inplace=True)

    if hardcoded_play_id is not None:
        row_data = df[df["playId"] == hardcoded_play_id].iloc[0]
    else:
        rows: int = df.shape[0]
        while True:
            row: int = random.randint(0, rows - 1)
            row_data = df.iloc[row]

            if os.path.exists(os.path.join(SS_CACHE_DIR, f"sporty-video/{row_data["playId"]}.mp4")):
                break


    game_pk: np.int64 = row_data["gamePk"].astype(np.int64)
    play_id: str = row_data["playId"]
    # seconds
    play_length: np.float64 = row_data["playLength"]
    # print(f"Chose {play_id}")

    open_command_csv_path = f"../data/2026/raw/gloveball_tracks/{game_pk}.csv.gz"
    oc_df: pd.DataFrame = pd.read_csv(open_command_csv_path, compression="gzip")

    sz_rows = sz_df.loc[(sz_df["game_pk"] == game_pk) & (sz_df["play_id"] == play_id)]
    sz: StrikeZoneEntry = StrikeZoneEntry()
    if not sz_rows.empty:
        sz_row = sz_rows.iloc[0]
        sz.x_left = np.float64(sz_row.x_left)
        sz.x_right = np.float64(sz_row.x_right)
        sz.y_top = np.float64(sz_row.y_top)
        sz.y_bottom = np.float64(sz_row.y_bottom)

    mp4_path: str = os.path.join(SS_CACHE_DIR, f"sporty-video/{play_id}.mp4")

    mp4: cv2.VideoCapture = cv2.VideoCapture(mp4_path)

    # frames / seconds
    mp4_framerate: np.float64 = np.float64(mp4.get(cv2.CAP_PROP_FPS))
    # frames
    mp4_frames: np.int64 = np.int64(mp4.get(cv2.CAP_PROP_FRAME_COUNT))
    # seconds
    mp4_length: np.float64 = mp4_frames / mp4_framerate
    initial_guess_release_mp4_frame: np.int64 = np.round((mp4_length - play_length) * mp4_framerate).astype(np.int64)

    baseball_center = oc_df[(oc_df["game_pk"] == game_pk) & (oc_df["play_id"] == play_id) & (oc_df["frame_idx"] >= 0)].sort_values(by="frame_idx")[["baseball_center_x", "baseball_center_y"]].to_numpy()
    best_score = 0.0
    best_score_offset = 0
    for offset in range(-5, 15 + 1):
        score = average_baseball_score_across_frames(mp4, initial_guess_release_mp4_frame.astype(int) + offset, baseball_center)
        # print(f"offset {offset}: {score}")
        if score > best_score:
            best_score = score
            best_score_offset = offset

    release_mp4_frame: np.int64 = initial_guess_release_mp4_frame + best_score_offset

    baseball_center_df = oc_df[(oc_df["game_pk"] == game_pk) & (oc_df["play_id"] == play_id)]
    while True:
        choice = random.random()
        if choice < 0.4:
            rand = np.int64(random.randint(-120, 0))
        else:
            rand = np.int64(random.randint(-15, 35))
        mp4_frame: np.int64 = rand + release_mp4_frame
        if mp4_frame < 0 or mp4_frame >= mp4_frames:
            continue
        frame_idx: np.int64 = mp4_frame - release_mp4_frame
        if is_good_sample(baseball_center_df, frame_idx):
            break


    glove_center_x, glove_center_y, baseball_center_x, baseball_center_y = get_oc_data_for_frame(oc_df, game_pk, play_id, frame_idx)

    origin: str = "sampler"

    # for v in range(0, mp4_frames - 1):
    #     draw_img_for_frame(mp4, np.int64(v), get_oc_data_for_frame(oc_df, game_pk, play_id, v - release_mp4_frame), sz, f"frame_{v:03}.png", redborder=v == release_mp4_frame)

    frame_width, frame_height = mp4.get(cv2.CAP_PROP_FRAME_WIDTH), mp4.get(cv2.CAP_PROP_FRAME_HEIGHT)

    write_raw_img_for_frame(mp4, mp4_frame, f"../dataset/ball/images/train/play_{play_id}.frame_{mp4_frame.astype(int):03}.png")
    if not np.isnan(baseball_center_x):
        with open(f"../dataset/ball/labels/train/play_{play_id}.frame_{mp4_frame.astype(int):03}.txt", 'w') as f:
            f.write(f"0 {(baseball_center_x.astype(float) / frame_width):.6f} {(baseball_center_y.astype(float) / frame_height):.6f} {(BASEBALL_WIDTH / frame_width):.6f} {(BASEBALL_HEIGHT / frame_height):.6f}")
    # draw_img_for_frame(mp4, mp4_frame, get_oc_data_for_frame(oc_df, game_pk, play_id, frame_idx), sz, f"../frames/play_{play_id}.frame_{mp4_frame.astype(int):03}.png")

    mp4.release()

    return f"{game_pk},{play_id},\"{mp4_path}\",{mp4_frame},{frame_idx},{release_mp4_frame},{glove_center_x},{glove_center_y},{baseball_center_x},{baseball_center_y},{sz.x_left},{sz.y_top},{sz.x_right},{sz.y_bottom},{origin}"

def is_row_nan(baseball_center_df: pd.DataFrame, frame_idx: int) -> bool | None:
    rows = baseball_center_df[baseball_center_df["frame_idx"] == frame_idx][["baseball_center_x", "baseball_center_y"]].to_numpy()
    if len(rows) == 0:
        return None
    else:
        x, y = rows[0]
        return np.isnan(x) or np.isnan(y)

def is_good_sample(baseball_center_df: pd.DataFrame, frame_idx: int) -> bool:
    N = 2

    rows = baseball_center_df[baseball_center_df["frame_idx"] == frame_idx]
    if rows.empty:
        return True
    else:
        res = is_row_nan(baseball_center_df, frame_idx)
        if res is None or not res:
            return True

        for frame_idx in chain(range(frame_idx - N, frame_idx), range(frame_idx + 1, frame_idx + 1 + N)):
            res = is_row_nan(baseball_center_df, frame_idx)
            if res is not None and not res:
                return False

        return True


def average_baseball_score_across_frames(mp4: cv2.VideoCapture, release_frame: int, baseball_center_by_frame: np.ndarray) -> float:
    sum = 0.0
    n = 0
    mp4.set(cv2.CAP_PROP_POS_FRAMES, release_frame - 1)
    _, previous_frame = mp4.read()
    px, py = baseball_center_by_frame[0]
    for idx, row in enumerate(baseball_center_by_frame):
        _, frame = mp4.read()
        x, y = row
        if np.isnan(x) or np.isnan(y):
            px = x
            py = y
            previous_frame = frame
            continue
        sum += baseball_score(previous_frame, frame, x, y, px, py)
        px = x
        py = y
        previous_frame = frame
        n += 1
    return sum / n

def lightness(pixels: np.ndarray) -> np.ndarray:
    Cmax = np.max(pixels, axis=2)
    Cmin = np.min(pixels, axis=2)
    return (Cmax + Cmin) / 2

def baseball_score(previous_frame, frame: cv2.typing.MatLike, x_f: np.float64, y_f: np.float64, px_f: np.float64, py_f: np.float64) -> float:
    R = 5

    x = np.round(x_f).astype(int)
    y = np.round(y_f).astype(int)
    dist = np.nan_to_num(np.sqrt((px_f - x) ** 2 + (py_f - y) ** 2), nan=0)
    previous_pixels = previous_frame[y - R : y + (R + 1), x - R : x + (R + 1)].astype(np.float64) / 255.0
    pixels = frame[y - R : y + (R + 1), x - R : x + (R + 1)].astype(np.float64) / 255.0
    pixel_lightness = lightness(pixels)
    previous_pixel_lightness = lightness(previous_pixels)
    diff = np.abs(pixel_lightness - previous_pixel_lightness)
    score = min(dist, R * 2 * np.sqrt(2)) * (pixel_lightness + diff)
    return np.mean(score).astype(float)

def get_oc_data_for_frame(oc_df: pd.DataFrame, game_pk: np.int64, play_id: str, frame_idx: np.int64):
    oc_rows = oc_df[(oc_df["game_pk"] == game_pk) & (oc_df["play_id"] == play_id) & (oc_df["frame_idx"] == frame_idx)]
    glove_center_x: np.float64 = np.nan if oc_rows.empty else oc_rows.iloc[0].glove_center_x
    glove_center_y: np.float64 = np.nan if oc_rows.empty else oc_rows.iloc[0].glove_center_y
    baseball_center_x: np.float64 = np.nan if oc_rows.empty else oc_rows.iloc[0].baseball_center_x
    baseball_center_y: np.float64 = np.nan if oc_rows.empty else oc_rows.iloc[0].baseball_center_y
    return glove_center_x, glove_center_y, baseball_center_x, baseball_center_y

def write_raw_img_for_frame(mp4: cv2.VideoCapture, mp4_frame: np.int64, filename: str):
    mp4.set(cv2.CAP_PROP_POS_FRAMES, mp4_frame.astype(int))
    _, frame = mp4.read()

    cv2.imwrite(filename, frame)

def draw_img_for_frame(mp4: cv2.VideoCapture, mp4_frame: np.int64, oc_data, sz: StrikeZoneEntry, filename: str, redborder: bool = False):
    glove_center_x, glove_center_y, baseball_center_x, baseball_center_y = oc_data

    mp4.set(cv2.CAP_PROP_POS_FRAMES, mp4_frame.astype(int))
    _, frame = mp4.read()

    if not np.isnan(glove_center_x) and not np.isnan(glove_center_y):
        cv2.circle(frame, (np.round(glove_center_x).astype(int), np.round(glove_center_y).astype(int)), 20, (0, 0, 255), 1)
    if not np.isnan(baseball_center_x) and not np.isnan(baseball_center_y):
        cv2.circle(frame, (np.round(baseball_center_x).astype(int), np.round(baseball_center_y).astype(int)), 10, (255, 255, 255), 1)
    if not np.isnan(sz.x_left):
        cv2.rectangle(frame, (np.round(sz.x_left).astype(int), np.round(sz.y_top).astype(int)), (np.round(sz.x_right).astype(int), np.round(sz.y_bottom).astype(int)), (0, 255, 255), 1)
    if redborder:
        cv2.rectangle(frame, (0, 0), (int(mp4.get(cv2.CAP_PROP_FRAME_WIDTH)) - 1, int(mp4.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 1), (0, 0, 255), 1)
    cv2.imwrite(filename, frame)

out = "game_pk,play_id,mp4_path,mp4_frame,frame_idx,release_mp4_frame,glove_center_x,glove_center_y,baseball_center_x,baseball_center_y,sz_x_left,sz_y_top,sz_x_right,sz_y_bottom,source"

def random_date(start, end):
    start: datetime.date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end: datetime.date = datetime.datetime.strptime(end, "%Y-%m-%d")
    delta = end - start
    delta_seconds = delta.days * 24 * 60 * 60 + delta.seconds
    random_second = random.randint(0, delta_seconds)
    return start + datetime.timedelta(seconds=random_second)

for _ in tqdm(range(5)):
    date = random_date("2026-04-01", "2026-08-13")
    date_string = date.strftime("%Y-%m-%d")
    res = get_row_for_sample(date_string)
    if res is not None:
        out += f"\n{res}"

f = open("../dataset/ball/rows.csv", 'w')
f.write(out)
f.close()

# print(get_row_for_sample("2026-08-13", hardcoded_play_id="14ebe9b0-efba-3d6c-bbba-65abe7bc3658")) # offset = 9
# print(get_row_for_sample("2026-08-13", hardcoded_play_id="64fa094b-16f6-331f-9ef0-eda20063dd62")) # offset = 5
# print(get_row_for_sample("2026-08-13", hardcoded_play_id="014f13fc-ea72-3fc6-971b-23a112e121a2")) # offset = 4
# print(get_row_for_sample("2026-08-13", hardcoded_play_id="62aaacc2-f590-38c9-a0ff-0e06efb6c5d3")) # offset = 1
