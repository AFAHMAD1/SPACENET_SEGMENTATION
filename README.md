# SpaceNet Segmentation

Semantic segmentation of satellite imagery into **land**, **road** and **building** classes, built on the SpaceNet 2 (AOI 3 – Paris) dataset with PyTorch.

## Overview

Raw SpaceNet pan-sharpened RGB tiles come with vector labels (building footprints and road centerlines with speed limits). This project converts those vectors into a 3-class raster mask for each tile, which a segmentation model (built with `segmentation-models-pytorch`) can then be trained on.

| Class | ID | Source |
|-------|----|--------|
| Land | 0 | background |
| Road | 1 | road centerlines buffered by estimated lane width (3.5 m per lane, clamped to 3–20 m) |
| Building | 2 | rasterized building footprints |

## Project structure

```
src/
  config.py        paths, class IDs, road-buffer and scaling parameters
  make_masks.py    builds 3-class masks and 8-bit image tiles from raw data
  check_masks.py   saves image/mask overlays for visual QA
notebooks/
  01_download.ipynb   data download (placeholder)
  02_make_masks.ipynb mask generation walkthrough
  03_train.ipynb      model training (placeholder)
  04_evaluate.ipynb   evaluation (placeholder)
results/mask_check/   sample overlays used to verify mask quality
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Dataset

The data is **not** stored in this repo (several GB). It comes from the public SpaceNet datasets (AOI 3 – Paris):

- SpaceNet on AWS (Registry of Open Data): https://registry.opendata.aws/spacenet/
- SpaceNet website / challenge descriptions: https://spacenet.ai/datasets/
- Bucket: `s3://spacenet-dataset/` (anonymous access), e.g.
  ```bash
  aws s3 ls --no-sign-request s3://spacenet-dataset/spacenet/
  aws s3 cp --no-sign-request s3://spacenet-dataset/spacenet/<challenge>/tarballs/<AOI_3_Paris archive> data/raw/
  ```
  Building footprints come from SpaceNet 2 (`SN2_buildings`) and road centerlines from SpaceNet 3 (`SN3_roads`); both cover AOI 3 – Paris. Extract the archives under `data/raw/`.

## Getting started

1. Download the data (see above) into `data/raw/`.
2. Generate masks: `python src/make_masks.py`
3. Sanity-check them: `python src/check_masks.py`
4. Build and split the dataset (`src/build_dataset.py`, `src/split.py`), train (`src/train.py`, or `notebooks/03_train_kaggle.ipynb` on Kaggle) and evaluate (`src/evaluate.py`). Metrics are in `results/metrics.md`.

## Stack

PyTorch, segmentation-models-pytorch, albumentations, rasterio, geopandas, shapely.

## Status

Data preparation and mask generation are in place. Download, training and evaluation notebooks are still placeholders.
