"""
DHARA — AOI DEM PREPROCESSING PIPELINE
=====================================

Processes ONE merged AOI DEM (UTM projected) and generates:

1. Slope
2. Aspect
3. Curvature
4. Flow Accumulation
5. Topographic Wetness Index (TWI)

INPUT REQUIREMENTS:
- AOI_DEM_Merged_UTM.tif
- CRS MUST be EPSG:32646
- Resolution should be 30m

OUTPUTS:
- AOI_slope.tif
- AOI_aspect.tif
- AOI_curvature.tif
- AOI_flow_accumulation.tif
- AOI_twi.tif

Install dependencies:
pip install rasterio pysheds scipy numpy
"""

import os
import numpy as np
import rasterio
from pysheds.grid import Grid
from scipy.ndimage import sobel
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

# CHANGE THIS TO YOUR DEM LOCATION
DEM_PATH = r"C:\Users\admin\Desktop\CNN_Flood\DEM\AOI_DEM_Merged_UTM.tif"

# OUTPUT FOLDER
OUTPUT_DIR = r"C:\Users\admin\Desktop\CNN_Flood\Final Stack"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CELL_SIZE = 30.0
NODATA = -9999.0

# =========================================================
# OUTPUT PATHS
# =========================================================

SLOPE_PATH = os.path.join(OUTPUT_DIR, "AOI_slope.tif")
ASPECT_PATH = os.path.join(OUTPUT_DIR, "AOI_aspect.tif")
CURVATURE_PATH = os.path.join(OUTPUT_DIR, "AOI_curvature.tif")
FLOW_ACC_PATH = os.path.join(OUTPUT_DIR, "AOI_flow_accumulation.tif")
TWI_PATH = os.path.join(OUTPUT_DIR, "AOI_twi.tif")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def save_raster(array, reference_path, output_path):
    with rasterio.open(reference_path) as src:
        meta = src.meta.copy()

    meta.update({
        "dtype": "float32",
        "count": 1,
        "nodata": NODATA
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array.astype("float32"), 1)


def load_dem(dem_path):
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")

        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan

    return dem


# =========================================================
# TERRAIN DERIVATIVES
# =========================================================

def compute_slope_aspect(dem, cell_size=30.0):

    dz_dx = sobel(np.nan_to_num(dem), axis=1) / (8.0 * cell_size)
    dz_dy = sobel(np.nan_to_num(dem), axis=0) / (8.0 * cell_size)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    aspect = np.degrees(np.arctan2(-dz_dy, dz_dx))
    aspect = (aspect + 360) % 360

    return slope_deg, aspect


def compute_curvature(dem, cell_size=30.0):

    d2z_dx2 = sobel(sobel(np.nan_to_num(dem), axis=1), axis=1) / (cell_size**2)
    d2z_dy2 = sobel(sobel(np.nan_to_num(dem), axis=0), axis=0) / (cell_size**2)

    curvature = -(d2z_dx2 + d2z_dy2)

    return curvature


def compute_flow_and_twi(dem_path, slope_deg):

    print("\n[INFO] Initializing pysheds grid...")

    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    print("[INFO] Filling pits...")
    pit_filled = grid.fill_pits(dem)

    print("[INFO] Filling depressions...")
    flooded = grid.fill_depressions(pit_filled)

    print("[INFO] Resolving flats...")
    conditioned = grid.resolve_flats(flooded)

    print("[INFO] Computing flow direction...")
    fdir = grid.flowdir(conditioned)

    print("[INFO] Computing flow accumulation...")
    acc = grid.accumulation(fdir)

    flow_acc = np.array(acc, dtype="float32")

    # Convert to area
    acc_area = flow_acc * (CELL_SIZE ** 2)
    acc_area = np.clip(acc_area, 1.0, None)

    print("[INFO] Computing TWI...")

    slope_rad = np.radians(slope_deg)

    tan_slope = np.tan(slope_rad)
    tan_slope = np.where(tan_slope < 1e-6, 1e-6, tan_slope)

    twi = np.log(acc_area / tan_slope)

    return flow_acc, twi


# =========================================================
# MAIN PROCESSING
# =========================================================

def main():

    print("=" * 60)
    print(" DHARA — AOI DEM PREPROCESSING PIPELINE ")
    print("=" * 60)

    print("\n[STEP 1] Loading DEM...")
    dem = load_dem(DEM_PATH)

    print("[✓] DEM loaded")

    # -----------------------------------------------------

    print("\n[STEP 2] Computing slope and aspect...")

    slope, aspect = compute_slope_aspect(dem, CELL_SIZE)

    save_raster(slope, DEM_PATH, SLOPE_PATH)
    save_raster(aspect, DEM_PATH, ASPECT_PATH)

    print("[✓] Slope saved")
    print("[✓] Aspect saved")

    # -----------------------------------------------------

    print("\n[STEP 3] Computing curvature...")

    curvature = compute_curvature(dem, CELL_SIZE)

    save_raster(curvature, DEM_PATH, CURVATURE_PATH)

    print("[✓] Curvature saved")

    # -----------------------------------------------------

    print("\n[STEP 4] Computing flow accumulation and TWI...")

    try:

        flow_acc, twi = compute_flow_and_twi(DEM_PATH, slope)

        save_raster(flow_acc, DEM_PATH, FLOW_ACC_PATH)
        save_raster(twi, DEM_PATH, TWI_PATH)

        print("[✓] Flow accumulation saved")
        print("[✓] TWI saved")

    except Exception as e:

        print(f"\n[ERROR] Flow/TWI failed: {e}")
        print("Try:")
        print("pip install pysheds --upgrade")

    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print(" PROCESSING COMPLETE ")
    print("=" * 60)

    print("\nGenerated Outputs:")

    print(f"\n1. {SLOPE_PATH}")
    print(f"2. {ASPECT_PATH}")
    print(f"3. {CURVATURE_PATH}")
    print(f"4. {FLOW_ACC_PATH}")
    print(f"5. {TWI_PATH}")

    print("\nNext Step:")
    print("→ Export JRC flood labels from Google Earth Engine")
    print("→ Align labels")
    print("→ CNN training")


# =========================================================

if __name__ == "__main__":

    try:
        import pysheds
        import rasterio
        import scipy

    except ImportError as e:

        print(f"\nMissing dependency: {e}")
        print("\nRun:")
        print("pip install rasterio pysheds scipy numpy")

        exit(1)

    main()