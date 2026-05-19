import os
import glob
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import earth2grid
from run_mars_dlesym import HEALPixUNet

# --- CONFIGURATION ---
LEVEL = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = 'mars_data/mars_reanalysis/'
MODEL_PATH = f"best_model_dlesym_level{LEVEL}.pth"
ROLLOUT_STEPS = 24  # 24 hours

def hpx_to_latlon(data_hpx_faces, level=LEVEL, nlat=180, nlon=360, device="cpu"):
    data_flat = data_hpx_faces.reshape(-1).to(device).float()
    src_hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
    tgt_ll = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    regridder = earth2grid.get_regridder(src_hpx, tgt_ll).to(device).float()
    data_ll = regridder(data_flat)
    return data_ll.reshape(nlat, nlon).cpu().numpy()

def main():
    print(f"Starting {ROLLOUT_STEPS}-hour rollout at HEALPix Level {LEVEL}...")
    file_list = sorted(glob.glob(DATA_PATH + '*.nc'))
    ds = xr.open_dataset(file_list[0])
    ds_min = xr.open_dataset('min_vals_mars_real.nc')
    ds_max = xr.open_dataset('max_vals_mars_real.nc')
    
    # Preprocessing
    ds_norm = (ds - ds_min) / (ds_max - ds_min)
    u_interp = ds_norm['U'].interp(latu=ds_norm.lat)
    ds_norm['U'] = u_interp.drop_vars('latu').fillna(0.0)
    v_interp = ds_norm['V'].interp(lonv=ds_norm.lon)
    ds_norm['V'] = v_interp.drop_vars('lonv').fillna(0.0)
    ds_norm['T'] = ds_norm['T'].fillna(0.0)

    # Conversion setup
    nlat, nlon = ds.sizes["lat"], ds.sizes["lon"]
    src = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    hpx = earth2grid.healpix.Grid(level=LEVEL, pixel_order=earth2grid.healpix.XY())
    regrid = earth2grid.get_regridder(src, hpx).to("cpu").float()
    
    def get_hpx_sample(t_idx):
        t_data = []
        for var in ['T', 'U', 'V']:
            v = torch.from_numpy(ds_norm[var].isel(time=t_idx).values).float()
            h = regrid(v.reshape(-1, nlat, nlon))
            t_data.append(h.reshape(28, 12, 2**LEVEL, 2**LEVEL))
        return torch.cat(t_data, dim=0)

    print("Initial state conversion...")
    current_state = get_hpx_sample(0).unsqueeze(0).to(DEVICE)
    
    print("Loading model...")
    model = HEALPixUNet(84, 84).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    rollout_preds = []
    ground_truth = []
    
    print("Executing rollout...")
    with torch.no_grad():
        for t in range(ROLLOUT_STEPS):
            # Predict next step
            current_state = model(current_state)
            rollout_preds.append(current_state.squeeze(0).cpu())
            # Get actual next step for comparison
            ground_truth.append(get_hpx_sample(t+1))
            if (t+1) % 6 == 0: print(f"  Step {t+1}/{ROLLOUT_STEPS} complete")

    print("Calculating metrics...")
    rmses = []
    for t in range(ROLLOUT_STEPS):
        # Calculate RMSE on normalized temperature (first 28 channels)
        mse = F.mse_loss(rollout_preds[t][:28], ground_truth[t][:28])
        rmses.append(torch.sqrt(mse).item())

    # Plot Rollout Error
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, ROLLOUT_STEPS + 1), rmses, marker='o', linewidth=2)
    plt.title(f"Mars Digital Twin Rollout Stability (Level {LEVEL})", fontsize=14)
    plt.xlabel("Forecast Horizon [hours]", fontsize=12)
    plt.ylabel("RMSE (Normalized Temperature)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig("rollout_stability_level6.png", dpi=300)
    print("Rollout stability plot saved as rollout_stability_level6.png")

    # Final visual check at hour 24
    print("Plotting final rollout state (T+24)...")
    var_idx = 0 # surface T
    true_ll = hpx_to_latlon(ground_truth[-1][var_idx], level=LEVEL)
    pred_ll = hpx_to_latlon(rollout_preds[-1][var_idx], level=LEVEL)

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    extent = [0, 360, -90, 90]
    im0 = axs[0].imshow(true_ll, cmap="inferno", extent=extent, origin='lower', aspect='auto')
    axs[0].set_title("EMARS Ground Truth (T+24h)", fontsize=16)
    axs[0].set_xlabel("Longitude [°E]")
    axs[0].set_ylabel("Latitude [°N]")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred_ll, cmap="inferno", extent=extent, origin='lower', aspect='auto')
    axs[1].set_title(f"DLESyM 24h Forecast (Level {LEVEL})", fontsize=16)
    axs[1].set_xlabel("Longitude [°E]")
    axs[1].set_ylabel("Latitude [°N]")
    plt.colorbar(im1, ax=axs[1])

    plt.tight_layout()
    plt.savefig(f"rollout_comparison_24h_level{LEVEL}.png", dpi=300)
    print(f"24h comparison plot saved as rollout_comparison_24h_level{LEVEL}.png")

if __name__ == "__main__":
    main()
