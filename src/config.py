from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RGB_DIR = RAW / "AOI_3_Paris_Train" / "RGB-PanSharpen"
BUILDING_DIR = RAW / "AOI_3_Paris_Train" / "geojson" / "buildings"
ROAD_DIR = RAW / "AOI_3_Paris" / "geojson_roads_speed"
IMAGES_DIR = ROOT / "data" / "images"
MASKS_DIR = ROOT / "data" / "masks"
TILES_CSV = ROOT / "data" / "tiles.csv"
CHECK_DIR = ROOT / "results" / "mask_check"

# Classes
CLASS_NAMES = {0: "land", 1: "road", 2: "building"}
LAND, ROAD, BUILDING = 0, 1, 2

# Roads
METRIC_CRS = "EPSG:32631"  # UTM 31N, used only for buffering
LANE_WIDTH_M = 3.5
DEFAULT_LANES = 2
MIN_WIDTH_M, MAX_WIDTH_M = 3.0, 20.0

# Image scaling
STRETCH_PCT = (2, 98)
N_SAMPLE_TILES = 100
SEED = 42
