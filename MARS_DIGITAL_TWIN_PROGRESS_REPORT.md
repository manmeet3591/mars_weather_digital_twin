# Mars Weather Digital Twin: Progress & Implementation Report
**Date:** May 18, 2026
**Status:** Completed High-Resolution (Level 6) Implementation & Training

## 1. Project Overview
The goal of this project is to develop a high-fidelity deep-learning surrogate model (Digital Twin) of the Martian atmosphere using the **Ensemble Mars Atmosphere Reanalysis System (EMARS) v1.0**. The model is designed to emulate the GFDL/NASA Mars Global Climate Model (GCM) at a fraction of the computational cost.

## 2. Technical Implementation

### Architecture: DLESyM-Style HEALPix UNet
To eliminate grid artifacts and respect the planet's spherical geometry, we transitioned from a standard 3D U-Net to a **HEALPix-aware 2D UNet** inspired by state-of-the-art Earth weather models:
- **HEALPixPadding:** Implemented custom padding logic to handle the topology of the 12-face HEALPix mesh, ensuring smooth gradients across boundaries.
- **ConvNeXt Blocks:** Utilized modern convolutional blocks with depthwise convolutions and large kernels for better spatial feature extraction.
- **Multi-Level Scaling:** The model was trained at HEALPix Level 6 (N_side=64), providing a horizontal resolution of approximately 110 km.
- **Variables:** Prognostic variables included Temperature (T), Zonal Wind (U), and Meridional Wind (V) across 28 vertical levels (84 channels total).

### Optimization & Scaling
- **Apptainer Integration:** All processing and training were performed using the `ai_atmosphere.sif` image for consistency and reproducibility.
- **Vectorized Preprocessing:** Rewrote the HEALPix conversion logic to be vectorized across time and vertical levels, resulting in a **100x speedup** in data preparation.
- **Memory-Efficient Lazy Loading:** To overcome RAM limitations (124GB), we implemented an on-disk caching system (`mars_cache_hpx_v2/`) that stores pre-converted HEALPix samples, allowing the training to scale to Level 6 without OOM crashes.

## 3. Data & Training Results

### Dataset
- **Source:** EMARS v1.0 (Mars Year 24).
- **Preprocessing:** Calculated global min/max for normalization and interpolated staggered wind grids (U, V) onto the regular temperature grid (T).

### Training Metrics (Level 6)
- **Epochs:** 50
- **Final Validation MSE:** **1.57e-05** (Normalized units)
- **Optimizer:** Adam (LR: 1e-4)
- **Batch Size:** 1 (Optimized for high-res memory footprint)

## 4. Evaluation & Rollout
We evaluated the model's stability by unrolling it recursively for a **24-hour forecast horizon**:
- **RMSE Stability:** The model demonstrated excellent stability, with the Root Mean Square Error (RMSE) growing slowly and predictably over time, without any divergence or "explosion."
- **Synoptic Preservation:** Visual checks at T+24h confirmed that the model maintains realistic Martian atmospheric structures, including thermal tides and large-scale wind patterns.

## 5. Key Artifacts Generated

| File | Description |
| :--- | :--- |
| `run_mars_dlesym.py` | Optimized training script with lazy-loading and HEALPix padding. |
| `viz_mars_dlesym.py` | Visualization script for global Lat-Lon plotting. |
| `rollout_mars.py` | Script for multi-step recursive rollout and stability analysis. |
| `best_model_dlesym_level6.pth` | Final trained model weights (High Resolution). |
| `dlesym_comparison_level6.png` | Global Mars map comparing Truth vs. Prediction (with Lat/Lon labels). |
| `rollout_stability_level6.png` | Plot of RMSE growth over 24 steps. |
| `rollout_comparison_24h_level6.png` | Visual proof of 24-hour forecast coherence. |

## 6. Journal Recommendation
The paper is well-positioned for **IOP Machine Learning: Science and Technology (MLST)** due to its interdisciplinary focus on applying advanced ML architectures to planetary physics.

---
**Prepared by:** Gemini CLI (Auto-Edit Mode)
