import glob

import pandas as pd
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union

import config as C
from make_masks import tile_id


def footprints(pattern):
    boxes, crs = [], set()
    for f in sorted(glob.glob(pattern)):
        with rasterio.open(f) as r:  # georeferencing only, no pixels read
            boxes.append(box(*r.bounds))
            crs.add(str(r.crs))
    return boxes, crs


def road_coverage():
    """% of each SN2 tile's area inside the union of all SN3 image footprints."""
    sn3, crs3 = footprints(str(C.RAW / "AOI_3_Paris" / "PS-MS" / "*.tif"))
    sn3_rgb, _ = footprints(str(C.RAW / "AOI_3_Paris" / "PS-RGB" / "*.tif"))
    print("SN3 footprints:", len(sn3), "(PS-MS) /", len(sn3_rgb), "(PS-RGB); CRS:", crs3)
    union = unary_union(sn3 + sn3_rgb)
    rows = []
    for p in sorted(C.RGB_DIR.glob("*.tif")):
        with rasterio.open(p) as r:
            b = box(*r.bounds)
        rows.append({"tile_id": tile_id(p),
                     "road_coverage_%": 100 * b.intersection(union).area / b.area})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = road_coverage()
    print(df["road_coverage_%"].describe().round(2))
    print((df["road_coverage_%"] >= C.MIN_ROAD_COVERAGE).sum(), "tiles >=", C.MIN_ROAD_COVERAGE, "%")
    print(df[df.tile_id == "AOI_3_Paris_img552"])
