import os
import numpy as np
import rasterio
from tensorflow.keras.models import load_model

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = r"C:\Users\admin\Desktop\CNN_Flood\Final Stack"

PATCH_SIZE = 15
HALF = PATCH_SIZE // 2

# =========================================================
# LOAD RASTER
# =========================================================

def load_raster(path):

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

    return arr

# =========================================================
# NORMALIZATION
# =========================================================

def normalize(arr):

    arr = np.nan_to_num(arr)

    arr_min = np.min(arr)
    arr_max = np.max(arr)

    return (arr - arr_min) / (arr_max - arr_min + 1e-8)

# =========================================================
# LOAD CHANNELS
# =========================================================

print("\nLoading rasters...")

dem = normalize(load_raster(os.path.join(DATA_DIR, "AOI_DEM_Merged_UTM.tif")))
slope = normalize(load_raster(os.path.join(DATA_DIR, "AOI_slope.tif")))
aspect = normalize(load_raster(os.path.join(DATA_DIR, "AOI_aspect.tif")))
curvature = normalize(load_raster(os.path.join(DATA_DIR, "AOI_curvature.tif")))
river = normalize(load_raster(os.path.join(DATA_DIR, "Distance_To_River.tif")))

# =========================================================
# MATCH DIMENSIONS
# =========================================================

min_rows = min(
    dem.shape[0],
    slope.shape[0],
    aspect.shape[0],
    curvature.shape[0],
    river.shape[0]
)

min_cols = min(
    dem.shape[1],
    slope.shape[1],
    aspect.shape[1],
    curvature.shape[1],
    river.shape[1]
)

dem = dem[:min_rows, :min_cols]
slope = slope[:min_rows, :min_cols]
aspect = aspect[:min_rows, :min_cols]
curvature = curvature[:min_rows, :min_cols]
river = river[:min_rows, :min_cols]

# =========================================================
# STACK
# =========================================================

stack = np.stack([
    dem,
    slope,
    aspect,
    curvature,
    river
], axis=-1)

print(f"\nStack Shape: {stack.shape}")

# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading trained CNN model...")

model = load_model(os.path.join(DATA_DIR, "flood_cnn_model.h5"))

print("Model loaded.")

# =========================================================
# OUTPUT MAP
# =========================================================

rows, cols, _ = stack.shape

susceptibility = np.zeros((rows, cols), dtype=np.float32)

print("\nGenerating susceptibility map...")

# =========================================================
# STRIDE SAMPLING
# =========================================================

STRIDE = 1

for i in range(HALF, rows - HALF, STRIDE):

    if i % 500 == 0:
        print(f"Processing row {i}/{rows}")

    patches = []
    coords = []

    for j in range(HALF, cols - HALF, STRIDE):

        patch = stack[
            i - HALF:i + HALF + 1,
            j - HALF:j + HALF + 1
        ]

        if patch.shape != (PATCH_SIZE, PATCH_SIZE, 5):
            continue

        patches.append(patch)
        coords.append((i, j))

    if len(patches) == 0:
        continue

    patches = np.array(patches, dtype=np.float32)

    preds = model.predict(patches, verbose=0)

    flood_probs = preds[:,1]

    for idx, (r, c) in enumerate(coords):

        susceptibility[r, c] = flood_probs[idx]

# =========================================================
# SAVE OUTPUT
# =========================================================

reference = os.path.join(DATA_DIR, "AOI_DEM_Merged_UTM.tif")

with rasterio.open(reference) as src:

    meta = src.meta.copy()

meta.update({
    "dtype": "float32",
    "count": 1
})

output_path = os.path.join(
    DATA_DIR,
    "Flood_Susceptibility_Mapv2.tif"
)

with rasterio.open(output_path, "w", **meta) as dst:

    dst.write(susceptibility, 1)

print("\nFlood susceptibility map saved.")
print(f"\nOutput: {output_path}")