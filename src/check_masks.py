import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import config as C

COLORS = {C.ROAD: (255, 255, 0), C.BUILDING: (255, 0, 0), C.IGNORE: (128, 128, 128)}


def overlay(img, mask, alpha=0.5):
    out = img.astype("float32").copy()
    for k, rgb in COLORS.items():
        m = mask == k
        out[m] = (1 - alpha) * out[m] + alpha * np.array(rgb, dtype="float32")
    return out.astype("uint8")


def pick_tiles(df, n=10):
    """Mix: 4 densest by total labelled %, 3 sparsest, 3 around the median."""
    df = df.dropna(subset=["road_pixels_%"])
    df = df.assign(lab=df["road_pixels_%"] + df["building_pixels_%"]).sort_values("lab")
    mid = len(df) // 2
    pick = list(df.tail(4).tile_id) + list(df.head(3).tile_id) + list(df.iloc[mid - 1:mid + 2].tile_id)
    return pick[:n]


def save_overlays(tile_ids, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tid in tile_ids:
        img = np.array(Image.open(C.IMAGES_DIR / f"{tid}.png"))
        mask = np.array(Image.open(C.MASKS_DIR / f"{tid}.png"))
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img)
        ax[1].imshow(overlay(img, mask))
        ax[0].set_title(tid)
        ax[1].set_title("road=yellow, building=red, no-data=gray")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{tid}.png", dpi=80)
        plt.close(fig)


def run():
    df = pd.read_csv(C.TILES_CSV)
    save_overlays(pick_tiles(df[df.kept]), C.CHECK_DIR)


def run_no_roads(n=3):
    """Overlays of kept tiles with 0 road pixels, to verify they really have no roads."""
    df = pd.read_csv(C.TILES_CSV)
    df = df[df.kept & (df["road_pixels_%"] == 0)]
    print(len(df), "kept tiles have 0% road pixels")
    save_overlays(list(df.sample(n, random_state=C.SEED).tile_id), C.NO_ROADS_DIR)


if __name__ == "__main__":
    run()
    run_no_roads()
