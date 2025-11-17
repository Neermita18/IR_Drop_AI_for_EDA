# IR Drop Prediction with ST-IA-UNet

Predict IR-drop (voltage drop) on VLSI layouts using a hybrid spatial–temporal UNet-style model trained on CircuitNet-derived data.

## Overview
This repository contains notebooks and scripts used to:
- Decompress CircuitNet archives into dataset files
- Construct train/test splits used in experiments
- Train a spatial + temporal UNet model for IR-drop prediction
- Visualize input channels, temporal frames, and predicted vs ground-truth IR-drop maps

The main experiment is in `ST_IA_UNet_IR_drop.ipynb`. The model implementation is in `model.py`.

## Repository structure
- `ST_IA_UNet_IR_drop.ipynb` — End-to-end experiment notebook (data loading, training loop, metrics, evaluation).
- `decompress_IR_drop/` — Scripts to decompress CircuitNet archives into usable dataset files.
- `create_train/` — Scripts to create the specific train/test dataset subsets used in experiments.
- `model.py` — PyTorch model definition:
  - DoubleConv2d, Down2d, Up2d
  - TemporalEncoder3D
  - IRDropUNet
- `visualize.ipynb` — Notebook to inspect / visualize input channels, intermediate features, and predictions.
- (Optional) `data/` — Recommended target for raw / decompressed / processed data.

## Model summary (see `model.py`)
- Spatial path: 2D UNet-style encoder/decoder built from DoubleConv2d, Down2d and Up2d blocks.
- Temporal path: `TemporalEncoder3D` — small 3D conv encoder that compresses temporal frames into a 2D feature map.
- Fusion: Temporal feature map is projected with a 1×1 convolution and concatenated to the spatial bottleneck before decoding.
- Output: single-channel IR-drop map. Forward supports optional interpolation to label size via `label_shape`.

Expected input shapes:
- spatial input: (B, in_spatial_ch, H, W) — default in_spatial_ch = 4
- temporal input: (B, temporal_ch, frames, H, W) — default temporal_ch = 1
- output: (B, 1, H_out, W_out) (upsampled to label size when `label_shape` is provided)

## Requirements
Minimum recommended:
- Python 3.8+
- PyTorch (choose CPU or CUDA build matching your hardware)
- jupyter
- numpy, pandas, matplotlib, seaborn
- scikit-learn
