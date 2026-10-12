import numpy as np
import pandas as pd
from PIL import Image

import check_masks
import config as C
import make_masks
from coverage import road_coverage


def clear_outputs():
    for d in (C.IMAGES_DIR, C.MASKS_DIR, C.CHECK_DIR):
        if d.exists():
            for f in d.glob("*.png"):
                f.unlink()


def drop_mostly_nodata(df):
    """Mark tiles with > MAX_NODATA % no-data as dropped and delete their PNGs."""
    bad = df.kept & (df["nodata_pixels_%"] > C.MAX_NODATA)
    df.loc[bad, "kept"] = False
    df.loc[bad, "drop_reason"] = f"nodata>{C.MAX_NODATA:g}%"
    for tid in df.tile_id[bad]:
        (C.IMAGES_DIR / f"{tid}.png").unlink(missing_ok=True)
        (C.MASKS_DIR / f"{tid}.png").unlink(missing_ok=True)
    return df


if __name__ == "__main__":
    cov = road_coverage()
    cov["kept"] = cov["road_coverage_%"] >= C.MIN_ROAD_COVERAGE
    cov["drop_reason"] = np.where(cov.kept, "", f"road_coverage<{C.MIN_ROAD_COVERAGE:g}%")
    keep = set(cov.tile_id[cov.kept])
    print("kept", len(keep), "of", len(cov))

    clear_outputs()
    stats = make_masks.run(keep_ids=keep)
    out = drop_mostly_nodata(cov.merge(stats, on="tile_id", how="left"))
    out.to_csv(C.TILES_CSV, index=False)
    keep = set(out.tile_id[out.kept])
    print("kept after no-data filter:", len(keep))

    # class percentages over all kept masks, 255 excluded
    counts = np.zeros(3, dtype="int64")
    for tid in keep:
        m = np.array(Image.open(C.MASKS_DIR / f"{tid}.png"))
        counts += np.bincount(m[m != C.IGNORE], minlength=3)[:3]
    pct = 100 * counts / counts.sum()
    print("class % (excl. 255):", dict(zip(C.CLASS_NAMES.values(), pct.round(2))))
    print("tiles with no-data pixels:", int((stats["nodata_pixels_%"] > 0).sum()), "of", len(stats))
    check_masks.run()
    check_masks.run_no_roads()
