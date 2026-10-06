import os
import sys
import pandas as pd

game_dir = sys.argv[1]
game_year = sys.argv[2]
print(game_dir)

pd = pd.concat([pd.read_csv(f'{game_dir}/{game}') for game in os.listdir(game_dir)])
print(pd)
print(pd.groupby("game_pk")[["sz_x_left", "sz_x_right", "sz_y_top", "sz_y_bottom"]].median())
