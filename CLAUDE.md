# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mars Weather Digital Twin — a deep-learning weather forecasting system for Mars, inspired by NVIDIA's DLESyM architecture. It trains neural networks on EMARS (European Mars Analysis Reanalysis System) reanalysis data to predict atmospheric state one time step ahead, then autoregressively rolls out multi-hour forecasts.

## Key Commands

```bash
# Activate the virtual environment
source venv_mars/bin/activate

# Train the DLESyM HEALPix UNet (production model, level 4 or 6)
python run_mars_dlesym.py

# Train the 3D UNet on real EMARS data (baseline)
python run_mars_real.py

# Run 24-hour autoregressive rollout evaluation
python rollout_mars.py

# Visualize single-step prediction comparison
python viz_mars_dlesym.py
```

Configuration constants (LEVEL, BATCH_SIZE, EPOCHS, LR, DATA_PATH) are set at the top of each script — there is no shared config file.

## Architecture

### Data Pipeline

1. **Source data**: EMARS NetCDF files in `mars_data/mars_reanalysis/` containing 3 atmospheric variables (T, U, V) across 28 vertical pressure levels, on a lat-lon grid.
2. **Grid interpolation**: U and V are on staggered grids (`latu`, `lonv`); they get interpolated to T's grid before processing.
3. **Normalization**: Min-max normalization using precomputed `min_vals_mars_real.nc` / `max_vals_mars_real.nc`.
4. **HEALPix regridding**: Lat-lon data is regridded to HEALPix faces using `earth2grid`. The result has shape `[time, 84, 12, nside, nside]` — 84 channels = 3 variables × 28 pressure levels, 12 HEALPix faces.
5. **Caching**: `run_mars_dlesym.py` caches individual HEALPix samples as `.pt` files in `mars_cache_hpx_v2/` to avoid re-converting each epoch. `MarsLazyDataset` loads them on demand.

### Model Variants

- **HEALPixUNet** (`run_mars_dlesym.py`, `dlesym_blocks.py`): The primary model. A 2D UNet that operates on `[B*12, C, H, W]` with HEALPix-aware padding (`HEALPixPadding`) that stitches neighboring faces at convolution boundaries. Uses ConvNeXt blocks with CappedGELU. Input/output: `[B, C, 12, H, W]` via FoldFaces/UnfoldFaces permutation.
- **UNet (3D)** (`run_mars_real.py`, notebook): A standard 3D UNet baseline treating the 12 HEALPix faces as a spatial dimension. Input/output: `[B, C, 12, H, W]`.

### HEALPix Face Layout

The padding logic in `HEALPixPadding` encodes the full 12-face adjacency graph: faces 0-3 (northern), 4-7 (equatorial), 8-11 (southern). Northern/southern faces require `rot90` transforms when borrowing data from neighbors. Equatorial corners use 50% blending (`_tl`, `_br`).

### Tensor Shape Convention

The data pipeline uses `[B, C, 12, H, W]` (channels before faces). The HEALPixUNet internally permutes to `[B, 12, C, H, W]` then folds to `[B*12, C, H, W]` for 2D convolutions.

### Training Pattern

All scripts follow: next-step prediction with MSE loss and Adam optimizer. Dataset pairs are `(data[t], data[t+1])`. Early stopping via patience counter. Best model saved as `best_model_*.pth`.

### Rollout Evaluation

`rollout_mars.py` feeds predictions back as input for up to 24 steps. Evaluates RMSE on normalized temperature across forecast horizon. `hpx_to_latlon()` regrids predictions back to lat-lon for visualization.

## Key Dependencies

- `earth2grid` — HEALPix ↔ lat-lon regridding (NVIDIA)
- `torch` (2.10+, CUDA 12.8)
- `xarray` — NetCDF data loading
- `wandb` — experiment tracking (used in `run_mars_real.py` and the notebook)

## Data Files

- `dummy_mars.nc` — synthetic test data for the notebook pipeline
- `min_vals_mars_real.nc` / `max_vals_mars_real.nc` — normalization bounds for real EMARS data
- `best_model_dlesym_level{4,6}.pth` — trained DLESyM model checkpoints
- `progress_dlesym_level{4,6}.json` — training loss history
