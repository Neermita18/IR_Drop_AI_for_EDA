# ST-IA-UNet: Spatial-Temporal Input-Attention U-Net for IR Drop Prediction

**ST-IA-UNet** is a deep learning model designed to predict static IR-drop (voltage drop) on VLSI chip layouts. By fusing 2D spatial power maps with 3D transient power sequences, it serves as a fast surrogate for expensive EDA simulations, instantly identifying voltage hotspots in complex power grids.

This model is trained on the **CircuitNet** dataset.

<img width="1774" height="516" alt="image" src="https://github.com/user-attachments/assets/a018c708-a334-47cb-8a10-d8fe18cdea58" />

---

## 1. Introduction
### What is IR Drop?
In modern semiconductor chip design, power is distributed through a massive grid of metal wires. As current ($I$) flows through these wires, the inherent electrical resistance ($R$) causes a voltage drop ($V = I \times R$).
* **The Problem:** If the voltage drops too much, transistors switch slower (timing violations) or fail completely.
* **The Goal:** Predict this voltage drop map instantly using Deep Learning, avoiding hours of slow SPICE simulations.

---

## 2. Data & Preprocessing
The model uses data derived from the **CircuitNet** dataset.

### Input Features
1. **Spatial Features (2D):** Four channels representing static power components.
   * `power_i` (Internal Power)  
   * `power_s` (Switching Power)  
   * `power_sca` (Leakage Power)  
   * `power_all` (Total Static Power)  
   * **Shape:** $(Batch, 4, 256, 256)$

2. **Temporal Features (3D):** A sequence of 20 time-steps showing dynamic power fluctuations.
   * `power_t` (Transient Power)  
   * **Shape:** $(Batch, 1, 20, 256, 256)$

### Preprocessing Pipeline
* **Resizing:** All maps standardized to $256 \times 256$.  
* **Normalization:** Min–Max normalization to $[0, 1]$.  
* **Log-Scaling:** Ground truth labels compressed using a logarithmic formula to handle the high dynamic range of voltage drops:

  $$y = \frac{\log(1 + x)}{\log(1 + x_{\max})}$$

---

## 3. Model Architecture
The model draws structural inspiration from CircuitNet but follows a **direct prediction architecture**. It uses:

### Architecture Components
1. **Spatial Encoder (2D):** Extracts geometric and structural features from static power maps.
2. **Temporal Encoder (3D):** Processes time-series power data.  
   `AdaptiveAvgPool3d` collapses the time dimension, summarizing temporal behavior at each spatial location.
3. **Fusion Bottleneck:** The spatial and temporal feature maps are concatenated and jointly decoded.

### Decoder
A standard U-Net decoder upsamples fused features and integrates skip connections from the encoder.  
The final layer predicts a **single** IR-drop map:

$$\text{Output Shape} = (Batch, 1, 256, 256)$$

---

## 4. Training Strategy
We used a **balanced loss function** to ensure both numerical accuracy and visual sharpness:

$$L_{Total} = L_{MAE} + \lambda \times L_{Gradient}$$

* **L1 Loss ($L_{MAE}$):** Ensures voltage values are numerically accurate.  
* **Gradient Difference Loss:** Preserves spatial sharpness by comparing pixel-wise derivatives.  
* **Weighting:** $\lambda = 0.1$, which sharpens power-grid edges without amplifying noise.

---

## 5. Results
* **Training Data:** 50 Samples  
* **Test Data:** 10 Samples  
* **Final Accuracy:** Average error of **~9.75 Volts** on a scale of 0–50 V.  
* **Visual Quality:**  
  The predictions reconstruct:
  * fine-grained vertical and horizontal power rails,  
  * high-drop "hotspot" regions,  
  demonstrating the effectiveness of spatial–temporal feature fusion.

---

## 6. Repository Structure
* `ST_IA_UNet_IR_drop.ipynb` — Main notebook (data loading, model, training, evaluation)  
* `decompress_IR_drop.py` — Reassembly/decompression for raw CircuitNet archives  
* `create_subset.py` — Script for generating train/test subsets  
* `irdropnet_final.pth` — Trained model weights  

---

## 7. Requirements
* Python 3.8+  
* PyTorch (CUDA recommended)  
* NumPy, Matplotlib  
* OpenCV (`opencv-python`)  

