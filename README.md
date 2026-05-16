# CNN-Based Flood Susceptibility Mapping in Assam Using Terrain-Derived Geospatial Factors

## Overview

This project presents a Convolutional Neural Network (CNN)-based geospatial framework for flood susceptibility mapping in selected flood-prone districts of Assam, India. The study integrates terrain-derived conditioning factors, GIS preprocessing, remote sensing datasets, and deep learning techniques to generate flood susceptibility maps.

The workflow combines Digital Elevation Model (DEM)-based terrain analysis with patch-based CNN classification to identify flood-prone and non-flood-prone regions. The project was developed as part of the B.Tech Geoinformatics curriculum at Netaji Subhas University of Technology (NSUT), West Campus.

---

## Study Area

The study focuses on the following flood-prone districts of Assam:

- Barpeta
- Dhemaji
- Lakhimpur
- Morigaon
- Nagaon

These districts are heavily influenced by the Brahmaputra River system and experience recurring seasonal flooding during the monsoon period.

---

## Objectives

The major objectives of this project are:

- Generate terrain-derived flood conditioning factors using DEM data
- Prepare flood-prone labels using JRC Global Surface Water data
- Build a CNN-based flood susceptibility classification framework
- Generate district-wise and overall flood susceptibility maps
- Develop a reproducible geospatial deep learning workflow

---

## Methodology

The overall workflow consists of the following stages:

1. DEM preprocessing
2. Terrain factor generation
3. River distance analysis
4. Flood label preparation using Google Earth Engine
5. Raster harmonization and alignment
6. Multi-channel raster stacking
7. Patch extraction for CNN input
8. CNN model training
9. Flood susceptibility map generation
10. Model evaluation and visualization

---

## Terrain Conditioning Factors

The following terrain-derived factors were used as CNN input channels:

- Elevation (DEM)
- Slope
- Aspect
- Curvature
- Distance to River

These factors were derived from a 30 m DEM and spatially harmonized before model training.

---

## Deep Learning Framework

A lightweight patch-based Convolutional Neural Network (CNN) architecture was used for binary flood susceptibility classification.

### CNN Workflow

```text
Raw DEM
   ↓
Terrain Preprocessing
   ↓
Flood Label Generation
   ↓
Raster Harmonization
   ↓
Patch Extraction (15×15)
   ↓
CNN Training
   ↓
Flood Susceptibility Mapping
```

### CNN Architecture

The CNN architecture consists of:

- Conv2D Layers
- MaxPooling Layers
- Flatten Layer
- Dense Fully Connected Layers
- Dropout Layer
- Softmax Output Layer

The model performs binary classification:

- 0 → Non-Flood-Prone
- 1 → Flood-Prone

---

## Google Earth Engine Flood Label Generation

Flood-prone labels were generated using the JRC Global Surface Water dataset in Google Earth Engine.

### File

```text
gee_flood_labels.js
```

### GEE Script

```javascript
var districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.and(
    ee.Filter.eq('ADM1_NAME', 'Assam'),
    ee.Filter.inList('ADM2_NAME',
      ['Barpeta','Dhemaji','Nagaon','Morigaon','Lakhimpur'])
  ));

var AOI = districts.geometry();

Map.centerObject(AOI, 8);

var labels = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence')
  .gt(20)
  .clip(AOI)
  .rename('flood');

Map.addLayer(labels, {min:0, max:1, palette:['black','cyan']}, 'Flood Labels');

Export.image.toDrive({
  image: labels,
  description: 'flood_labels_assam',
  folder: 'DHARA_Assam',
  fileNamePrefix: 'flood_labels_assam',
  scale: 30,
  region: AOI,
  maxPixels: 1e11
});
```

Pixels with water occurrence greater than 20% were considered flood-prone.

---

## Repository Structure

```text
assam-flood-susceptibility-cnn/
│
├── input/
│   ├── DEM/
│   ├── Rivers/
│   └── Flood_Labels/
│
├── processed_data/
│   ├── slope.tif
│   ├── aspect.tif
│   ├── curvature.tif
│   ├── river_distance.tif
│   └── raster_stack.npy
│
├── outputs/
│   ├── figures/
│   ├── susceptibility_maps/
│   └── evaluation/
│
├── gee_flood_labels.js
├── preprocess_dem.py
├── cnn_dataset.py
├── train_cnn.py
├── generate_map.py
│
├── flood_cnn_model.h5
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Workflow Execution

The workflow should be executed in the following order.

---

### Step 1 — Flood Label Generation

Run the Google Earth Engine script:

```javascript
gee_flood_labels.js
```

This generates flood-prone labels from the JRC Global Surface Water dataset.

Output:
- flood_labels_assam.tif

---

### Step 2 — DEM Preprocessing

Run:

```bash
python preprocess_dem.py
```

This script:
- preprocesses DEM data
- generates terrain factors
- creates slope, aspect, curvature, and river-distance rasters
- harmonizes raster projections and dimensions

Generated outputs:
- slope.tif
- aspect.tif
- curvature.tif
- river_distance.tif

---

### Step 3 — CNN Dataset Preparation

Run:

```bash
python cnn_dataset.py
```

This script:
- stacks raster layers
- extracts 15×15 spatial patches
- prepares train-test datasets
- converts raster data into CNN-ready format

Generated outputs:
- X_train.npy
- y_train.npy
- X_test.npy
- y_test.npy

---

### Step 4 — CNN Model Training

Run:

```bash
python train_cnn.py
```

This script:
- trains the CNN model
- evaluates classification performance
- saves trained model weights
- generates evaluation metrics

Generated outputs:
- flood_cnn_model.h5
- training_curves.png
- confusion_matrix.png

---

### Step 5 — Flood Susceptibility Map Generation

Run:

```bash
python generate_map.py
```

This script:
- loads trained CNN model
- predicts flood susceptibility
- generates final flood susceptibility raster maps

Generated outputs:
- Flood_Susceptibility_Map.tif

---

## Software and Libraries

### GIS Software

- QGIS 3.x
- Google Earth Engine

### Python Environment

- Python 3.10

### Major Python Libraries

- TensorFlow
- Keras
- NumPy
- Rasterio
- GDAL
- Scikit-learn
- Matplotlib
- Pandas
- OpenCV

---

## Installation

Install required Python packages using:

```bash
pip install -r requirements.txt
```

---

## requirements.txt

```text
tensorflow
keras
numpy
matplotlib
rasterio
gdal
scikit-learn
opencv-python
pandas
```

---

## Results

The trained CNN model achieved the following performance:

- Test Accuracy: 99.23%
- ROC-AUC Score: 0.9768

The generated susceptibility maps show that high flood-prone zones are concentrated near river corridors and low-elevation floodplain regions, consistent with the hydrological characteristics of Assam.

---

## Figures Generated

The workflow produces:

- DEM maps
- Slope maps
- Curvature maps
- River-distance maps
- Flood label maps
- CNN training curves
- Confusion matrix
- District-wise susceptibility maps
- Overall flood susceptibility map

---

## Reproducibility and Data Availability

The project workflow has been organized to support reproducibility.

The repository contains:
- preprocessing scripts
- CNN training scripts
- susceptibility generation workflow
- evaluation outputs

Large raster datasets, trained models, and processed outputs are available through cloud storage links.

### GitHub Repository

```text
[Insert GitHub Repository Link]
```

### Google Drive / Cloud Storage

```text
[Insert Google Drive Link]
```

To reproduce the workflow:

1. Download input datasets
2. Run preprocessing scripts
3. Generate raster stack
4. Train CNN model
5. Generate susceptibility maps

All raster layers must:
- share the same CRS
- have aligned raster dimensions
- be normalized before CNN training

---

## Limitations

Some limitations of the current study include:

- Limited flood inventory labels
- Class imbalance between flood and non-flood samples
- Absence of rainfall and temporal hydrological variables
- Static susceptibility analysis rather than real-time forecasting

---

## Future Scope

Possible future improvements include:

- Integration of Sentinel-1 SAR flood data
- Rainfall-based temporal flood modeling
- ConvLSTM architectures
- Attention-based deep learning models
- Multi-hazard susceptibility mapping
- State-wide flood susceptibility assessment

---

## References

1. J. Debnath et al., “Evaluating flood susceptibility in the Brahmaputra River Basin,” *Earth Systems and Environment*, 2023.

2. R. Bentivoglio et al., “Deep learning methods for flood mapping,” *Hydrology and Earth System Sciences*, 2022.

3. Y. Wang et al., “Flood susceptibility mapping using convolutional neural network frameworks,” *Journal of Hydrology*, 2020.

4. G. Zhao et al., “Urban flood susceptibility assessment based on convolutional neural networks,” *Journal of Hydrology*, 2020.

5. Y. O. Ouma and L. Omai, “Flood susceptibility mapping using image-based 2D-CNN deep learning,” *International Journal of Intelligent Systems*, 2023.

---

## Author

Paranjai G  
B.Tech Geoinformatics  
Netaji Subhas University of Technology (NSUT), West Campus

Project Supervisors:
- Dr. Navdeep Agrawal
- Dr. Sanjeev Kumar

---

## License

This project is released under the MIT License.
