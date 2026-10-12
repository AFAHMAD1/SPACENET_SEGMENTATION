import numpy as np
import pandas as pd
from PIL import Image

import config as C


def make_splits():
    df = pd.read_csv(C.TILES_CSV)
    ids = np.array(sorted(df.tile_id[df.kept]))
    rng = np.random.RandomState(C.SEED)
    rng.shuffle(ids)
    n_train = round(C.SPLIT_FRACS["train"] * len(ids))
    n_val = round(C.SPLIT_FRACS["val"] * len(ids))
    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * (len(ids) - n_train - n_val))
    out = pd.DataFrame({"tile_id": ids, "split": split}).sort_values("tile_id")
    out.to_csv(C.SPLITS_CSV, index=False)
    return out


def class_percentages(tile_ids):
    counts = np.zeros(3, dtype="int64")
    for tid in tile_ids:
        m = np.array(Image.open(C.MASKS_DIR / f"{tid}.png"))
        counts += np.bincount(m[m != C.IGNORE], minlength=3)[:3]
    return dict(zip(C.CLASS_NAMES.values(), (100 * counts / counts.sum()).round(2)))


if __name__ == "__main__":
    s = make_splits()
    for name, g in s.groupby("split"):
        print(name, len(g), class_percentages(g.tile_id))
