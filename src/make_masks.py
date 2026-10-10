import random

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm

import config as C


def tile_id(path):
    return path.stem.replace("RGB-PanSharpen_", "")


def load_roads():
    """All road features merged, buffered by road width, back in EPSG:4326."""
    frames = [gpd.read_file(f) for f in sorted(C.ROAD_DIR.glob("*.geojson"))]
    roads = gpd.GeoDataFrame(pd.concat([f for f in frames if len(f)], ignore_index=True),
                             crs=frames[0].crs)
    lanes = pd.to_numeric(roads["lane_number"], errors="coerce")
    lanes = lanes.where(lanes > 0, C.DEFAULT_LANES).fillna(C.DEFAULT_LANES)
    width = (lanes * C.LANE_WIDTH_M).clip(C.MIN_WIDTH_M, C.MAX_WIDTH_M)
    metric = roads.to_crs(C.METRIC_CRS)
    metric["geometry"] = metric.geometry.buffer(width / 2.0)
    return metric.to_crs(roads.crs)[["geometry"]]


def percentile_stretch(tiles, rng_seed=C.SEED):
    """Global per-band (lo, hi) from a random sample of tiles; zero (nodata) pixels ignored."""
    sample = random.Random(rng_seed).sample(tiles, min(C.N_SAMPLE_TILES, len(tiles)))
    px = []
    for p in sample:
        with rasterio.open(p) as r:
            a = r.read().reshape(3, -1)
        px.append(a[:, a.any(axis=0)])
    px = np.concatenate(px, axis=1)
    lo = np.percentile(px, C.STRETCH_PCT[0], axis=1)
    hi = np.percentile(px, C.STRETCH_PCT[1], axis=1)
    return lo, hi


def to_uint8(arr, lo, hi):
    out = (arr.astype("float32") - lo[:, None, None]) / (hi - lo)[:, None, None]
    return (np.clip(out, 0, 1) * 255).round().astype("uint8").transpose(1, 2, 0)


def build_mask(shape, transform, roads, buildings):
    mask = np.zeros(shape, dtype="uint8")
    if len(roads):
        mask = rasterize(((g, C.ROAD) for g in roads.geometry), out_shape=shape,
                         transform=transform, fill=0, dtype="uint8")
    if len(buildings):
        b = rasterize(((g, C.BUILDING) for g in buildings.geometry), out_shape=shape,
                      transform=transform, fill=0, dtype="uint8")
        mask[b == C.BUILDING] = C.BUILDING  # buildings win on overlap
    return mask


def run():
    C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    C.MASKS_DIR.mkdir(parents=True, exist_ok=True)
    tiles = sorted(C.RGB_DIR.glob("*.tif"))
    lo, hi = percentile_stretch(tiles)
    print("stretch lo", lo, "hi", hi)

    roads = load_roads()
    sindex = roads.sindex

    rows = []
    for p in tqdm(tiles):
        with rasterio.open(p) as r:
            bounds, transform, shape = r.bounds, r.transform, (r.height, r.width)
            tile_box = box(*bounds)
            hit = sindex.query(tile_box, predicate="intersects")
            if len(hit) == 0:
                continue  # only tiles with at least one road
            img = r.read()
        tid = tile_id(p)
        b_path = C.BUILDING_DIR / f"buildings_{tid}.geojson"
        buildings = gpd.read_file(b_path) if b_path.exists() else gpd.GeoDataFrame(geometry=[])
        mask = build_mask(shape, transform, roads.iloc[hit], buildings)
        Image.fromarray(to_uint8(img, lo, hi)).save(C.IMAGES_DIR / f"{tid}.png")
        Image.fromarray(mask).save(C.MASKS_DIR / f"{tid}.png")
        n = mask.size
        rows.append({"tile_id": tid,
                     "road_pixels_%": 100 * (mask == C.ROAD).sum() / n,
                     "building_pixels_%": 100 * (mask == C.BUILDING).sum() / n,
                     "land_pixels_%": 100 * (mask == C.LAND).sum() / n})
    df = pd.DataFrame(rows)
    df.to_csv(C.TILES_CSV, index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(len(df), "tiles")
    print(df[["land_pixels_%", "road_pixels_%", "building_pixels_%"]].mean().round(2))
