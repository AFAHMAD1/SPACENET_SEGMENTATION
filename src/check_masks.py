import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import config as C

COLORS = {C.ROAD: (255, 255, 0), C.BUILDING: (255, 0, 0)}


def overlay(img, mask, alpha=0.5):
    out = img.astype("float32").copy()
    for k, rgb in COLORS.items():
        m = mask == k
        out[m] = (1 - alpha) * out[m] + alpha * np.array(rgb, dtype="float32")
    return out.astype("uint8")


def pick_tiles(df, n=10):
    """Mix: 4 densest by total labelled %, 3 sparsest, 3 around the median."""
    df = df.assign(lab=df["road_pixels_%"] + df["building_pixels_%"]).sort_values("lab")
    mid = len(df) // 2
    pick = list(df.tail(4).tile_id) + list(df.head(3).tile_id) + list(df.iloc[mid - 1:mid + 2].tile_id)
    return pick[:n]


def run():
    C.CHECK_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.TILES_CSV)
    for tid in pick_tiles(df):
        img = np.array(Image.open(C.IMAGES_DIR / f"{tid}.png"))
        mask = np.array(Image.open(C.MASKS_DIR / f"{tid}.png"))
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img)
        ax[1].imshow(overlay(img, mask))
        ax[0].set_title(tid)
        ax[1].set_title("road=yellow, building=red")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(C.CHECK_DIR / f"{tid}.png", dpi=80)
        plt.close(fig)


if __name__ == "__main__":
    run()
