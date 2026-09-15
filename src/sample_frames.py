import datetime
import random
from typing import Optional

import pandas as pd
import numpy as np
import cv2
import os

SS_CACHE_DIR: str = "/Users/riley/Library/Caches/statcast-subsidiary"

sz_df = pd.read_csv("../data/2026/raw/strikezone_tracking.csv.gz", compression="gzip")

class StrikeZoneEntry:
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float

# todo: remove duplicates by pruning duplicate rows later
# game_pk,play_id,mp4_path,mp4_frame,frame_idx,release_mp4_frame,glove_center_x,glove_center_y,baseball_center_x,baseball_center_y,sz_x_left,sz_y_top,sz_x_right,sz_y_bottom
def get_row_for_sample(date: str, hardcoded_play_id: Optional[str] = None) -> Optional[str]:
    date: datetime.date = datetime.datetime.strptime(date, "%Y-%m-%d")
    play_ids_for_date_path: str = os.path.join(SS_CACHE_DIR, f"statcast-play-ids/{date.year}/{date.strftime("%Y-%m-%d")}.csv")
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
    print(f"chose {play_id}")

    open_command_csv_path = f"../data/2026/raw/gloveball_tracks/{game_pk}.csv.gz"
    oc_df: pd.DataFrame = pd.read_csv(open_command_csv_path, compression="gzip")

    sz_row = sz_df.loc[(sz_df["game_pk"] == game_pk) & (sz_df["play_id"] == play_id)].iloc[0]
    sz: StrikeZoneEntry = StrikeZoneEntry()
    sz.x_left = sz_row.x_left
    sz.x_right = sz_row.x_right
    sz.y_top = sz_row.y_top
    sz.y_bottom = sz_row.y_bottom

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
        print(f"offset: {offset} = {score}")
        if score > best_score:
            best_score = score
            best_score_offset = offset

    release_mp4_frame: np.int64 = initial_guess_release_mp4_frame + best_score_offset

    # mp4_frame: np.int64 = min(max(rand + release_mp4_frame, np.int64(0)), np.int64(mp4_frames - 1))
    # frame_idx: np.int64 = mp4_frame - release_mp4_frame

    # glove_center_x, glove_center_y, baseball_center_x, baseball_center_y = get_oc_data_for_frame(oc_df, game_pk, play_id, frame_idx)

    origin: str = "sampler"

    for v in range(0, mp4_frames - 1):
        draw_img_for_frame(mp4, np.int64(v), get_oc_data_for_frame(oc_df, game_pk, play_id, v - release_mp4_frame), sz_row, f"frame_{v:03}.png", redborder=v == release_mp4_frame)

    # return f"{game_pk},{play_id},\"{mp4_path}\",{mp4_frame},{frame_idx},{release_mp4_frame},{glove_center_x},{glove_center_y},{baseball_center_x},{baseball_center_y},{sz.x_left},{sz.y_top},{sz.x_right},{sz.y_bottom},{origin}"
    return ""

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
    diff = np.mean(np.abs(pixel_lightness, previous_pixel_lightness), axis=1)
    score = pixel_lightness + min(dist, R * 2 * np.sqrt(2)) * diff
    return np.mean(score)

def get_oc_data_for_frame(oc_df: pd.DataFrame, game_pk: np.int64, play_id: str, frame_idx: np.int64):
    oc_rows = oc_df[(oc_df["game_pk"] == game_pk) & (oc_df["play_id"] == play_id) & (oc_df["frame_idx"] == frame_idx)]
    glove_center_x: np.float64 = np.nan if oc_rows.empty else oc_rows.iloc[0].glove_center_x
    glove_center_y: np.float64 = np.nan if oc_rows.empty else oc_rows.iloc[0].glove_center_y
    baseball_center_x: np.float64 = np.nan if oc_rows.empty else oc_rows.iloc[0].baseball_center_x
    baseball_center_y: np.float64 = np.nan if oc_rows.empty else oc_rows.iloc[0].baseball_center_y
    return glove_center_x, glove_center_y, baseball_center_x, baseball_center_y

def draw_img_for_frame(mp4: cv2.VideoCapture, mp4_frame: np.int64, oc_data, sz_row, filename: str, redborder: bool = False):
    glove_center_x, glove_center_y, baseball_center_x, baseball_center_y = oc_data

    sz_x_left = sz_row.x_left
    sz_x_right = sz_row.x_right
    sz_y_top = sz_row.y_top
    sz_y_bottom = sz_row.y_bottom

    mp4.set(cv2.CAP_PROP_POS_FRAMES, mp4_frame.astype(int))
    _, frame = mp4.read()

    if not np.isnan(glove_center_x) and not np.isnan(glove_center_y):
        cv2.circle(frame, (np.round(glove_center_x).astype(int), np.round(glove_center_y).astype(int)), 20, (255, 0, 0), 1)
    if not np.isnan(baseball_center_x) and not np.isnan(baseball_center_y):
        cv2.circle(frame, (np.round(baseball_center_x).astype(int), np.round(baseball_center_y).astype(int)), 10, (255, 255, 255), 1)
    cv2.rectangle(frame, (np.round(sz_x_left).astype(int), np.round(sz_y_top).astype(int)), (np.round(sz_x_right).astype(int), np.round(sz_y_bottom).astype(int)), (0, 255, 255), 1)
    if redborder:
        cv2.rectangle(frame, (0, 0), (int(mp4.get(cv2.CAP_PROP_FRAME_WIDTH)) - 1, int(mp4.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 1), (255, 0, 0), 1)
    cv2.imwrite(filename, frame)

# print(get_row_for_sample("2026-08-13", hardcoded_play_id="14ebe9b0-efba-3d6c-bbba-65abe7bc3658")) # offset = 9
# print(get_row_for_sample("2026-08-13", hardcoded_play_id="64fa094b-16f6-331f-9ef0-eda20063dd62")) # offset = 5
# print(get_row_for_sample("2026-08-13", hardcoded_play_id="014f13fc-ea72-3fc6-971b-23a112e121a2")) # offset = 4
