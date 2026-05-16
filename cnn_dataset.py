import os
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split

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

labels = load_raster(os.path.join(DATA_DIR, "flood_labels_aligned.tif"))

print("All rasters loaded.")
# =========================================================
# MATCH DIMENSIONS
# =========================================================

min_rows = min(
    dem.shape[0],
    slope.shape[0],
    aspect.shape[0],
    curvature.shape[0],
    river.shape[0],
    labels.shape[0]
)

min_cols = min(
    dem.shape[1],
    slope.shape[1],
    aspect.shape[1],
    curvature.shape[1],
    river.shape[1],
    labels.shape[1]
)

dem = dem[:min_rows, :min_cols]
slope = slope[:min_rows, :min_cols]
aspect = aspect[:min_rows, :min_cols]
curvature = curvature[:min_rows, :min_cols]
river = river[:min_rows, :min_cols]
labels = labels[:min_rows, :min_cols]

# =========================================================
# STACK CHANNELS
# =========================================================

stack = np.stack([
    dem,
    slope,
    aspect,
    curvature,
    river
], axis=-1)

print(f"\nStack Shape: {stack.shape}")
print(f"Labels Shape: {labels.shape}")


# =========================================================
# PATCH EXTRACTION
# =========================================================

X = []
y = []

rows, cols = labels.shape

print("\nSampling random patches...")

# =========================================================
# RANDOM SAMPLE SETTINGS
# =========================================================

NUM_SAMPLES = 30000

# =========================================================
# VALID PIXELS
# =========================================================

valid_pixels = np.argwhere((labels == 0) | (labels == 1))

print(f"Valid Pixels Found: {len(valid_pixels)}")

# Random selection
indices = np.random.choice(
    len(valid_pixels),
    size=min(NUM_SAMPLES, len(valid_pixels)),
    replace=False
)

sampled_pixels = valid_pixels[indices]

# =========================================================
# PATCH EXTRACTION
# =========================================================

for pixel in sampled_pixels:

    i, j = pixel

    # Boundary check
    if (
        i - HALF < 0 or
        i + HALF >= rows or
        j - HALF < 0 or
        j + HALF >= cols
    ):
        continue

    patch = stack[
        i - HALF:i + HALF + 1,
        j - HALF:j + HALF + 1
    ]

    if patch.shape != (PATCH_SIZE, PATCH_SIZE, 5):
        continue

    if np.isnan(patch).any():
        continue

    X.append(patch)
    y.append(labels[i, j])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"\nTotal Samples: {len(X)}")

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset Ready.")

print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")

# =========================================================
# SAVE DATASET
# =========================================================

np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)

np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

print("\nDataset saved successfully.")