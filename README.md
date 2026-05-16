# CNN-Based Flood Susceptibility Mapping in Assam

## Overview

This project presents a CNN-based flood susceptibility mapping framework for selected flood-prone districts of Assam using terrain-derived geospatial factors and deep learning.

The workflow integrates GIS preprocessing, terrain analysis, raster harmonization, patch extraction, and Convolutional Neural Network (CNN) classification to generate flood susceptibility maps.

The study focuses on the following districts of Assam:

- Barpeta
- Dhemaji
- Lakhimpur
- Morigaon
- Nagaon

---

## Objectives

The main objectives of this project are:

- Generate terrain-derived flood conditioning factors from DEM data
- Prepare flood-prone labels using JRC Global Surface Water data
- Build a patch-based CNN model for flood susceptibility classification
- Generate district-wise and overall flood susceptibility maps
- Develop a reproducible geospatial deep learning workflow

---

## Methodology

The overall workflow consists of:

1. DEM preprocessing
2. Terrain factor generation
3. River distance analysis
4. Flood label preparation
5. Raster harmonization and alignment
6. Multi-channel raster stacking
7. Patch extraction (15×15)
8. CNN model training
9. Susceptibility map generation
10. Model evaluation

---

## Terrain Conditioning Factors

The following factors were used as CNN input channels:

- Elevation (DEM)
- Slope
- Aspect
- Curvature
- Distance to River

---

## Deep Learning Architecture

A lightweight patch-based CNN architecture was used for binary classification of flood-prone and non-flood-prone areas.

### Model Structure

- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Flatten
- Dense Layer
- Dropout
- Output Layer (Softmax)

---

## Software and Libraries

### GIS Tools

- QGIS
- Google Earth Engine

### Python Libraries

- TensorFlow
- Keras
- NumPy
- Rasterio
- GDAL
- Scikit-learn
- Matplotlib

---

## Dataset Sources

### DEM Data
- SRTM DEM

### Flood Labels
- JRC Global Surface Water Dataset

### River Data
- HydroSHEDS / OpenStreetMap River Network

---

## Results

The CNN model achieved:

- Test Accuracy: 99.23%
- ROC-AUC Score: 0.9768

The generated susceptibility maps show strong flood-prone patterns near river corridors and low-lying terrain regions.

---

## Repository Structure

```bash
data/                  # Input raster datasets
outputs/               # Generated maps and figures
scripts/               # Python scripts
models/                # Saved CNN model
docs/                  # Project report
```

---

## Reproducibility

The workflow can be reproduced using the preprocessing scripts and Python environment described in the report.

All raster layers should be:

- spatially aligned
- normalized
- resampled to common resolution
- projected to the same CRS

before CNN training.

---

## Future Improvements

Potential future improvements include:

- Sentinel-1 flood integration
- Rainfall-based temporal modeling
- ConvLSTM architectures
- Multi-hazard susceptibility mapping
- Attention-based CNN models

---

## Author

Paranjai Gusaria 
B.Tech Geoinformatics  
Netaji Subhas University of Technology (NSUT)

---

## License

This project is released under the MIT License.
