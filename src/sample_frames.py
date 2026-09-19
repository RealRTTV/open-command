import datetime
import random
import sys

from tqdm import tqdm
from itertools import chain
from typing import Optional, List

import pandas as pd
import numpy as np
import cv2
import os
import platform

SS_CACHE_DIR: str = "/Users/riley/Library/Caches/statcast-subsidiary" if platform.system() == "Darwin" else "/home/riley/.cache/statcast-subsidiary"
BASEBALL_WIDTH: int = 20
BASEBALL_HEIGHT: int = 20
DIR = "val"

sz_df = pd.read_csv("../data/2026/raw/strikezone_tracking.csv.gz", compression="gzip")

class StrikeZoneEntry:
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float

    def __init__(self):
        self.x_left = np.nan
        self.x_right = np.nan
        self.y_top = np.nan
        self.y_bottom = np.nan

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
        row: int = random.randint(0, rows - 1)
        row_data = df.iloc[row]

        if not os.path.exists(os.path.join(SS_CACHE_DIR, f"sporty-video/{row_data["playId"]}.mp4")):
            return None


    game_pk: int = row_data["gamePk"].astype(int)
    play_id: str = row_data["playId"]
    # print(f"Chose {play_id}")

    open_command_csv_path = f"../data/2026/raw/gloveball_tracks/{game_pk}.csv.gz"
    if not os.path.exists(open_command_csv_path):
        return None
    oc_df: pd.DataFrame = pd.read_csv(open_command_csv_path, compression="gzip")

    sz_rows = sz_df.loc[(sz_df["game_pk"] == game_pk) & (sz_df["play_id"] == play_id)]
    sz: StrikeZoneEntry = StrikeZoneEntry()
    if not sz_rows.empty:
        sz_row = sz_rows.iloc[0]
        sz.x_left = sz_row.x_left
        sz.x_right = sz_row.x_right
        sz.y_top = sz_row.y_top
        sz.y_bottom = sz_row.y_bottom

    mp4_path: str = os.path.join(SS_CACHE_DIR, f"sporty-video/{play_id}.mp4")

    mp4: cv2.VideoCapture = cv2.VideoCapture(mp4_path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])

    old_frame_width, old_frame_height = int(mp4.get(cv2.CAP_PROP_FRAME_WIDTH)), int(mp4.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_size = (min(old_frame_width, old_frame_height) // 32) * 32
    x0 = (old_frame_width - frame_size) // 2
    x1 = old_frame_width - x0
    y0 = (old_frame_height - frame_size) // 2
    y1 = old_frame_height - y0
    screenspace = x0, x1, y0, y1

    crop_x_shift = max(0, (old_frame_width - frame_size) // 2)
    crop_y_shift = max(0, (old_frame_height - frame_size) // 2)

    crop_shift = np.array([crop_x_shift, crop_y_shift])

    sz.x_left -= crop_x_shift
    sz.x_right -= crop_x_shift
    sz.y_top -= crop_y_shift
    sz.y_bottom -= crop_y_shift

    # frames / seconds
    # mp4_framerate: float = mp4.get(cv2.CAP_PROP_FPS)
    # frames
    mp4_frames: int = int(mp4.get(cv2.CAP_PROP_FRAME_COUNT))
    # seconds
    # mp4_length: float = mp4_frames / mp4_framerate
    initial_guess_release_mp4_frame: int = 180

    baseball_center: np.ndarray = oc_df[(oc_df["game_pk"] == game_pk) & (oc_df["play_id"] == play_id) & (oc_df["frame_idx"] >= 0)].sort_values(by="frame_idx")[["baseball_center_x", "baseball_center_y"]].to_numpy() - crop_shift
    if baseball_center.size == 0:
        mp4.release()
        return None

    all_relevant_frames_starting_idx = initial_guess_release_mp4_frame - 15 - 1
    mp4.set(cv2.CAP_PROP_POS_FRAMES, all_relevant_frames_starting_idx)
    all_relevant_frames = [mp4.read()[1] for _ in range(all_relevant_frames_starting_idx, initial_guess_release_mp4_frame + 30 + baseball_center.size - 1)]
    scores = []
    best_score: float = 0.0
    best_score_offset = 0
    for offset in range(-15, 30 + 1):
        score = average_baseball_score_across_frames(all_relevant_frames, all_relevant_frames_starting_idx, initial_guess_release_mp4_frame + offset, baseball_center, crop_shift)
        if np.isnan(score):
            continue
        # print(f"offset {offset}: {score:.4f}")
        scores.append(score)
        if score > best_score:
            best_score = score
            best_score_offset = offset
    if len(scores) <= 1 or best_score not in scores:
        mp4.release()
        return None
    scores.remove(best_score)
    second_best_score = max(scores)
    if abs(second_best_score - best_score) < 0.05:
        mp4.release()
        return None

    # print(f"best offset {best_score_offset}: {best_score}")

    if best_score <= 0.0:
        mp4.release()
        return None

    release_mp4_frame: int = initial_guess_release_mp4_frame + best_score_offset

    baseball_center_df = oc_df[(oc_df["game_pk"] == game_pk) & (oc_df["play_id"] == play_id)]
    has_ball = baseball_center_df[~np.isnan(baseball_center_df.baseball_center_x)]
    max_frame_idx = 0 if has_ball.frame_idx.empty else int(has_ball.frame_idx.max())
    frame_idx: int = 0
    while True:
        choice = random.random()
        rand: int
        if choice < 0.4:
            rand = random.randint(-120, 0)
        else:
            rand = random.randint(-15, max_frame_idx + 7)
        mp4_frame: int = rand + release_mp4_frame
        if mp4_frame < 0 or mp4_frame >= mp4_frames:
            continue
        frame_idx: int = mp4_frame - release_mp4_frame
        if is_good_sample(baseball_center_df, frame_idx):
            break

    glove_center_x, glove_center_y, baseball_center_x, baseball_center_y = get_oc_data_for_frame(oc_df, game_pk, play_id, frame_idx, crop_shift)

    origin: str = "sampler"

    # for v in range(0, mp4_frames - 1):
    #     draw_img_for_frame(mp4, v, get_oc_data_for_frame(oc_df, game_pk, play_id, v - release_mp4_frame, crop_shift), sz, crop_shift, f"frame_{v:03}.png", screenspace, red_border=v == release_mp4_frame)

    write_raw_img_for_frame(mp4, mp4_frame, f"../dataset/ball/images/{DIR}/play_{play_id}.frame_{mp4_frame:03}.png", screenspace)
    if not np.isnan(baseball_center_x):
        with open(f"../dataset/ball/labels/{DIR}/play_{play_id}.frame_{mp4_frame:03}.txt", 'w') as f:
            f.write(f"0 {float(baseball_center_x / frame_size):.6f} {float(baseball_center_y / frame_size):.6f} {(BASEBALL_WIDTH / frame_size):.6f} {(BASEBALL_HEIGHT / frame_size):.6f}")
    # draw_img_for_frame(mp4, mp4_frame, get_oc_data_for_frame(oc_df, game_pk, play_id, frame_idx, crop_shift), sz, crop_shift, f"../frames/play_{play_id}.frame_{mp4_frame:03}.png", screenspace)

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
    MIN_NAN_SURROUNDING = 1

    rows = baseball_center_df[baseball_center_df["frame_idx"] == frame_idx]
    if rows.empty:
        return True
    else:
        res = is_row_nan(baseball_center_df, frame_idx)
        if res is None or not res:
            return True

        for frame_idx in chain(range(frame_idx - MIN_NAN_SURROUNDING, frame_idx), range(frame_idx + 1, frame_idx + 1 + MIN_NAN_SURROUNDING)):
            res = is_row_nan(baseball_center_df, frame_idx)
            if res is not None and not res:
                return False

        return True

def average_baseball_score_across_frames(all_relevant_frames: List[cv2.typing.MatLike], all_relevant_frames_starting_index: int, release_frame: int, baseball_center_by_frame: np.ndarray, crop_shift: np.ndarray) -> float:
    arr = []
    for idx, row in enumerate(baseball_center_by_frame):
        frame = all_relevant_frames[release_frame + idx - all_relevant_frames_starting_index]
        x, y = row
        if np.isnan(x) or np.isnan(y) or frame is None or frame.size == 0:
            continue
        arr.append(baseball_score(frame, x, y, crop_shift))

    return np.median(np.array(arr, dtype=float))

def lightness(pixels: np.ndarray) -> np.ndarray:
    c_max = np.max(pixels, axis=2)
    c_min = np.min(pixels, axis=2)
    return (c_max + c_min) / 2


SCORING_RADIUS = 8

def baseball_score(frame: cv2.typing.MatLike, x_f: float, y_f: float, crop_shift: np.ndarray) -> float:
    grid_y, grid_x = np.ogrid[-SCORING_RADIUS:SCORING_RADIUS + 1, -SCORING_RADIUS:SCORING_RADIUS + 1]
    distance = np.sqrt(grid_y ** 2 + grid_x ** 2)
    ball = distance <= SCORING_RADIUS
    lit = ball & (grid_y < 0)
    unlit = ball & (grid_y >= 0)
    background = ~ball
    target_lightness = background * 0.0 + lit * 1.0 + unlit * 0.6

    x = int(x_f + crop_shift[0] + 0.5)
    y = int(y_f + crop_shift[1] + 0.5)

    pixels = frame[y - SCORING_RADIUS: y + (SCORING_RADIUS + 1), x - SCORING_RADIUS: x + (SCORING_RADIUS + 1)].astype(float) / 255.0
    pixel_lightness = lightness(pixels)

    if len(pixel_lightness) != len(target_lightness):
        return 0.0

    score = np.corrcoef(pixel_lightness.ravel(), target_lightness.ravel())[0, 1]
    return np.mean(score).astype(float)

def get_oc_data_for_frame(oc_df: pd.DataFrame, game_pk: int, play_id: str, frame_idx: int, crop_shift: np.ndarray):
    oc_rows = oc_df[(oc_df["game_pk"] == game_pk) & (oc_df["play_id"] == play_id) & (oc_df["frame_idx"] == frame_idx)]
    row = None if oc_rows.empty else oc_rows.iloc[0]
    if row is None:
        return np.nan, np.nan, np.nan, np.nan
    else:
        return row.glove_center_x - crop_shift[0], row.glove_center_y - crop_shift[1], row.baseball_center_x - crop_shift[0], row.baseball_center_y - crop_shift[1]

def write_raw_img_for_frame(mp4: cv2.VideoCapture, mp4_frame: int, filename: str, screenspace):
    x0, x1, y0, y1 = screenspace
    mp4.set(cv2.CAP_PROP_POS_FRAMES, mp4_frame)
    _, frame = mp4.read()

    cv2.imwrite(filename, frame[y0:y1, x0:x1])

def draw_img_for_frame(mp4: cv2.VideoCapture, mp4_frame: int, oc_data, sz: StrikeZoneEntry, crop_shift: np.ndarray, filename: str, screenspace, red_border: bool = False):
    glove_center_x, glove_center_y, baseball_center_x, baseball_center_y = oc_data
    x0, x1, y0, y1 = screenspace

    mp4.set(cv2.CAP_PROP_POS_FRAMES, mp4_frame)
    _, frame = mp4.read()

    if not np.isnan(glove_center_x) and not np.isnan(glove_center_y):
        cv2.circle(frame, (int(glove_center_x + crop_shift[0]), int(glove_center_y + crop_shift[1])), 20, (0, 0, 255), 1)
    if not np.isnan(baseball_center_x) and not np.isnan(baseball_center_y):
        cv2.circle(frame, (int(baseball_center_x + crop_shift[0]), int(baseball_center_y + crop_shift[1])), 10, (255, 255, 255), 1)
    if not np.isnan(sz.x_left):
        cv2.rectangle(frame, (int(sz.x_left + crop_shift[0]), int(sz.y_top + crop_shift[1])), (int(sz.x_right + crop_shift[0]), int(sz.y_bottom + crop_shift[1])), (0, 255, 255), 1)
    if red_border:
        cv2.rectangle(frame, (0, 0), (int(mp4.get(cv2.CAP_PROP_FRAME_WIDTH)) - 1, int(mp4.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 1), (0, 0, 255), 1)
    cv2.imwrite(filename, frame[y0:y1, x0:x1])

def random_date(start, end):
    start: datetime.date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end: datetime.date = datetime.datetime.strptime(end, "%Y-%m-%d")
    delta = end - start
    delta_seconds = delta.days * 24 * 60 * 60 + delta.seconds
    random_second = random.randint(0, delta_seconds)
    return start + datetime.timedelta(seconds=random_second)

def main():
    f = open(f"../dataset/ball/rows_{DIR}.csv", 'a')
    if os.path.getsize(f"../dataset/ball/rows_{DIR}.csv") == 0:
        f.write("game_pk,play_id,mp4_path,mp4_frame,frame_idx,release_mp4_frame,glove_center_x,glove_center_y,baseball_center_x,baseball_center_y,sz_x_left,sz_y_top,sz_x_right,sz_y_bottom,source")
    try:
        n = int(sys.argv[1])
    except IndexError, ValueError:
        n = 10_000
    for _ in tqdm(range(n)):
        while True:
            date = random_date("2026-04-01", "2026-08-13")
            date_string = date.strftime("%Y-%m-%d")
            res = get_row_for_sample(date_string)
            if res is not None:
                f.write(f"\n{res}")
                break
    f.close()

main()

# print(get_row_for_sample("2026-08-13", hardcoded_play_id="14ebe9b0-efba-3d6c-bbba-65abe7bc3658")) # offset = 5
# print(get_row_for_sample("2026-08-13")) # offset = ?
# print(get_row_for_sample("2026-08-13", hardcoded_play_id="014f13fc-ea72-3fc6-971b-23a112e121a2")) # offset = 2
# print(get_row_for_sample("2026-08-13", hardcoded_play_id="62aaacc2-f590-38c9-a0ff-0e06efb6c5d3")) # offset = -2
