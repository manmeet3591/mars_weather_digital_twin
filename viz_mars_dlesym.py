import os
import glob
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import earth2grid
from run_mars_dlesym import HEALPixUNet

# --- CONFIGURATION ---
LEVEL = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = 'mars_data/mars_reanalysis/'
MODEL_PATH = f"best_model_dlesym_level{LEVEL}.pth"

def hpx_to_latlon(data_hpx_faces, level=LEVEL, nlat=180, nlon=360, device="cpu"):
    data_flat = data_hpx_faces.reshape(-1).to(device).float()
    src_hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
    tgt_ll = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    regridder = earth2grid.get_regridder(src_hpx, tgt_ll).to(device).float()
    data_ll = regridder(data_flat)
    return data_ll.reshape(nlat, nlon).cpu().numpy()

def main():
    print(f"Loading data for Level {LEVEL} rollout visualization...")
    file_list = sorted(glob.glob(DATA_PATH + '*.nc'))[:1]
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

    # Manual conversion for 2 samples (t0 for input, t1 for ground truth)
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

    print("Converting samples...")
    hpx_0 = get_hpx_sample(0)
    hpx_1 = get_hpx_sample(1)

    print("Loading model...")
    model = HEALPixUNet(84, 84).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Predicting...")
    with torch.no_grad():
        pred = model(hpx_0.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

    print("Plotting comparison...")
    var_idx = 0 # Surface Temperature
    true_ll = hpx_to_latlon(hpx_1[var_idx], level=LEVEL)
    pred_ll = hpx_to_latlon(pred[var_idx], level=LEVEL)

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    extent = [0, 360, -90, 90]
    
    im0 = axs[0].imshow(true_ll, cmap="inferno", extent=extent, origin='lower', aspect='auto')
    axs[0].set_title("EMARS Ground Truth (T, surface)", fontsize=16)
    axs[0].set_xlabel("Longitude [°E]", fontsize=12)
    axs[0].set_ylabel("Latitude [°N]", fontsize=12)
    plt.colorbar(im0, ax=axs[0], label="Normalized T")

    im1 = axs[1].imshow(pred_ll, cmap="inferno", extent=extent, origin='lower', aspect='auto')
    axs[1].set_title(f"DLESyM Prediction Level {LEVEL}", fontsize=16)
    axs[1].set_xlabel("Longitude [°E]", fontsize=12)
    axs[1].set_ylabel("Latitude [°N]", fontsize=12)
    plt.colorbar(im1, ax=axs[1], label="Normalized T")

    plt.tight_layout()
    plt.savefig(f"dlesym_comparison_level{LEVEL}.png", dpi=300)
    print(f"Plot saved as dlesym_comparison_level{LEVEL}.png")

if __name__ == "__main__":
    main()
