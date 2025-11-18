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
1.  **Spatial Features (2D):** Four channels representing static power components.
    * `power_i` (Internal Power)
    * `power_s` (Switching Power)
    * `power_sca` (Leakage Power)
    * `power_all` (Total Static Power)
    * *Shape:* $(Batch, 4, 256, 256)$
2.  **Temporal Features (3D):** A sequence of 20 time-steps showing dynamic power fluctuations.
    * `power_t` (Transient Power)
    * *Shape:* $(Batch, 1, 20, 256, 256)$

### Preprocessing Pipeline
Raw physical data cannot be fed directly into a neural network. We applied:
* **Resizing:** All maps standardized to $256 \times 256$.
* **Normalization:** Min-Max normalization to $[0, 1]$.
* **Log-Scaling:** Ground truth labels are compressed using a logarithmic formula to handle the high dynamic range of voltage drops (preventing the model from ignoring rare spikes):

    $$y = \frac{\log(1 + x)}{\log(1 + x_{\max})}$$

---

## 3. Model Architecture
The model uses **Input Attention Mechanism**, which solves the blurriness problem common in standard U-Nets. This is referenced from [CircuitNet's](https://circuitnet.github.io/tutorial/experiment_tutorial.html#ir-drop-prediction) official tutorial.

### Architecture Components
1.  **Spatial Encoder (2D):** A CNN extracting geometric features from the static power maps.
2.  **Temporal Encoder (3D):** A 3D CNN processing time-series data. It uses `AdaptiveAvgPool3d` to collapse the time dimension, effectively asking: *"What was the average or peak activity at this location across the entire time window?"*
3.  **Fusion Bottleneck:** The features from both encoders are concatenated, merging structural info (grid layout) with dynamic info (activity spikes).

### The "Input Attention" Trick
Instead of generating the IR drop image from scratch (which causes blur), the decoder predicts **4 Weight Maps**. These weights are multiplied element-wise with the original high-resolution **Input Spatial Maps**.

$$\text{Output} = \sum (\text{Input Spatial Maps} \times \text{Learned Weights})$$

This transfers the crisp, sharp power-grid geometry directly into the prediction.

---

## 4. Training Strategy
We used a **Balanced Loss Function** to ensure both numerical accuracy and visual sharpness.

$$L_{Total} = L_{MAE} + \lambda \times L_{Gradient}$$

* **L1 Loss ($L_{MAE}$):** Ensures accurate voltage values.
* **Gradient Difference Loss:** Penalizes blurriness by comparing the edges (derivatives) of the prediction against the ground truth.
* **Weighting:** We set $\lambda = 0.1$, a "Goldilocks" zone that sharpens grid lines without introducing noise artifacts.

---

## 5. Results
* **Training Data:** 50 Samples (CircuitNet Subset)
* **Test Data:** 10 Samples
* **Final Accuracy:** The model achieved an average error of **~9.75 Volts** on a scale of 0–50 V.
* **Visual Quality:** The model successfully reconstructs fine-grained power grid lines and correctly locates "hotspots" (yellow areas) where voltage drop is critical.

---

## 6. Repository Structure
* `ST_IA_UNet_IR_drop.ipynb` — Main experiment notebook (Data loading, Model definition, Training, Evaluation).
* `decompress_IR_drop.py` — Helper script to reassemble and decompress raw CircuitNet tarballs.
* `create_subset.py` — Script to generate the fixed train/test split used in this experiment.
* `irdropnet_final.pth` — Trained model weights.

## 7. Requirements
* Python 3.8+
* PyTorch (CUDA recommended)
* NumPy, Matplotlib
* OpenCV (`opencv-python`)
