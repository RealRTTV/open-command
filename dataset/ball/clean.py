import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("rows.csv")
df.dropna(subset=["release_mp4_frame"], inplace=True)

total = df.release_mp4_frame.count()
for n in range(180 - 30, 180 + 30):
    num = df[df.release_mp4_frame == n].release_mp4_frame.count()
    print(f"{n}: {100.0 * num / total:.2f}%")

mid = 180
dev = 15

print(f"[{mid - dev}, {mid + dev}]: {100.0 * df[(df.release_mp4_frame >= mid - dev) & (df.release_mp4_frame <= mid + dev)].release_mp4_frame.count() / total:.2f}%")

# df[df.release_mp4_frame < 250].hist(column="release_mp4_frame")
# plt.show()
