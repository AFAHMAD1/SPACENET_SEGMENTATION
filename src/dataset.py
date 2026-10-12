from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

import config as C

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP = 512


def get_transforms(train):
    if train:
        spatial = [A.RandomCrop(CROP, CROP),
                   A.HorizontalFlip(p=0.5),
                   A.VerticalFlip(p=0.5),
                   A.RandomRotate90(p=1.0),
                   # light photometric jitter; touches the image only, never the mask
                   A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)]
    else:
        spatial = [A.CenterCrop(CROP, CROP)]
    return A.Compose(spatial + [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


class SpaceNetDataset(Dataset):
    """Image/mask pairs for one split. Mask values: 0 land, 1 road, 2 building, 255 ignore."""

    def __init__(self, split, transform=None, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else C.ROOT / "data"  # holds images/, masks/, splits.csv
        df = pd.read_csv(self.data_dir / "splits.csv")
        self.ids = list(df.tile_id[df.split == split])
        self.transform = transform if transform is not None else get_transforms(split == "train")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        tid = self.ids[i]
        image = np.array(Image.open(self.data_dir / "images" / f"{tid}.png").convert("RGB"))
        mask = np.array(Image.open(self.data_dir / "masks" / f"{tid}.png"))
        out = self.transform(image=image, mask=mask)
        return out["image"].float(), out["mask"].long()  # (3,512,512) float, (512,512) long
