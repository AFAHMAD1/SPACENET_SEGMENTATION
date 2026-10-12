import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as C
from dataset import IMAGENET_MEAN, IMAGENET_STD, SpaceNetDataset

PALETTE = {C.LAND: (0, 0, 0), C.ROAD: (255, 255, 0), C.BUILDING: (255, 0, 0), C.IGNORE: (128, 128, 128)}


def colorize(mask):
    out = np.zeros((*mask.shape, 3), dtype="uint8")
    for k, rgb in PALETTE.items():
        out[mask == k] = rgb
    return out


def denorm(x):
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return ((x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")


if __name__ == "__main__":
    ds = SpaceNetDataset("train")
    print("train tiles:", len(ds), "| val:", len(SpaceNetDataset("val")), "| test:", len(SpaceNetDataset("test")))
    idx = np.random.RandomState(C.SEED).choice(len(ds), 4, replace=False)
    fig, ax = plt.subplots(4, 2, figsize=(8, 16))
    for r, i in enumerate(idx):
        x, y = ds[i]
        print(ds.ids[i], tuple(x.shape), x.dtype, tuple(y.shape), y.dtype, "mask values:", y.unique().tolist())
        ax[r, 0].imshow(denorm(x))
        ax[r, 0].set_title(ds.ids[i])
        ax[r, 1].imshow(colorize(y.numpy()))
        ax[r, 1].set_title("road=yellow, building=red, ignore=gray")
        for a in ax[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(C.ROOT / "results" / "dataset_check.png", dpi=80)

    # val is deterministic: same sample twice must match
    v = SpaceNetDataset("val")
    a, b = v[0], v[0]
    print("val deterministic:", bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])), "| val shapes", tuple(a[0].shape), tuple(a[1].shape))
