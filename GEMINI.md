# ROCSTOR Workspace: AI for Earth & Atmosphere

Welcome to the **ROCSTOR** workspace. This directory is a centralized repository for advanced AI research and development in Earth sciences, meteorology, and oceanography. It contains multiple independent yet interconnected projects focusing on deep learning-based forecasting, data assimilation, and foundation models.

## 🌍 Workspace Overview

This workspace is organized into several key domains:
- **Atmospheric Modeling:** AI-driven weather forecasting and National Weather Service (NWS) discussion generation.
- **Ocean & Land Modeling:** Global prediction systems for ocean states and land surface processes using 3D architectures and HEALPix grids.
- **Data Assimilation:** Modular frameworks for integrating observations with generative priors.
- **Foundation Models & Embeddings:** High-resolution geospatial embeddings and multi-modal pre-training.

## 📂 Key Projects

### 🌊 Ocean & Land Modeling
- **`ai_ocean_model/`**: 3D UNet system for global ocean state (Potential Temp, Salinity) prediction using HEALPix grids.
- **`ai_land_model/`**: S2S (Sub-seasonal to Seasonal) land surface modeling.
- **`aircast_hongkong/`**: Regional high-resolution urban weather modeling for the RUMI intercomparison.

### 🌪️ Atmospheric & Weather
- **`AI-NWS/`**: An "AI Meteorologist" system using GRPO and LLMs to generate professional Forecast Discussions.
- **`graphcast/`**: DeepMind's GraphCast implementation/integration for global weather forecasting.
- **`beat_stormscope/`**: Lightning and storm prediction (StormScope).
- **`insat_nowcast/`**: Satellite-based nowcasting using INSAT data.

### 🧪 Data Assimilation & Utilities
- **`DeepAssimilate/`**: Modular framework using diffusion models for generative data assimilation.
- **`alphaearth_embeddings/`**: Large-scale processing of 30m global geospatial embeddings.
- **`stationbench/`**: Benchmarking suite for weather station data.
- **`SMAP-HydroBlocks_postprocessing/`**: Hydrological modeling post-processing.

## 🛠️ Shared Infrastructure & Tools

### Containerization (Apptainer/Singularity)
Most projects utilize **Apptainer** for environment consistency.
- **Images:** `.sif` files are often located within project roots or in `apptainer_cache/`.
- **Definitions:** `.def` files define the software stack (PyTorch, Xarray, earth2grid, etc.).
- **Common Images:** `apptainer_al_land.sif` and `apptainer_ndui_production.sif` are shared across multiple land/ocean tasks.

### Core Technologies
- **Frameworks:** PyTorch, Hugging Face Diffusers, Unsloth.
- **Data Handling:** Xarray, Dask, Zarr, NetCDF, GeoTIFF.
- **Spatial Representations:** HEALPix (for global grids), regular Lat-Lon (for regional/urban).
- **Experiment Tracking:** Weights & Biases (WandB) is the standard for logging metrics and checkpoints.

## 📜 General Development Conventions

1.  **Project-Specific Context:** Many subdirectories contain their own `GEMINI.md` or `README.md`. Always check the subdirectory first for specific build/run instructions.
2.  **Data Access:** Large datasets are typically streamed (e.g., ARCO-ERA5 from Google Cloud Zarr) or stored in scratch directories. Verify paths in `normalization.json` or config scripts.
3.  **Reproducibility:** Prefer running scripts via Apptainer:
    ```bash
    apptainer exec --nv <image>.sif python <script>.py
    ```
4.  **Hardware:** Assume NVIDIA GPU availability (`--nv`) for training and inference.

## 🚀 Navigation Guide

- To explore a specific project, navigate to its directory and look for a `GEMINI.md`.
- Use `ls -R` or `glob` patterns to find specific model checkpoints (`.pth`) or training logs.
- Temporary files and cache are located in `apptainer_tmp/` and `apptainer_cache/`.
