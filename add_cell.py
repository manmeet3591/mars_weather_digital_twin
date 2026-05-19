import json

with open('mars_weather_dt.ipynb', 'r') as f:
    nb = json.load(f)

new_code = """
import matplotlib.pyplot as plt
import earth2grid
import torch
import numpy as np

def hpx_to_latlon(data_hpx_faces, level=6, nlat=180, nlon=360, device="cpu"):
    \"\"\"
    Converts a HEALPix tensor of shape [12, N, N] back to a Lat-Lon grid.
    \"\"\"
    nside = 2**level
    # Flatten faces to 1D array for earth2grid
    data_flat = data_hpx_faces.reshape(-1).to(device)
    
    src_hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
    tgt_ll = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    
    regridder = earth2grid.get_regridder(src_hpx, tgt_ll).to(device)
    data_ll = regridder(data_flat)
    return data_ll.reshape(nlat, nlon).cpu().numpy()

def plot_global_rollout(pred_seq, true_seq, var="T", level_index=0):
    \"\"\"
    Plot predicted vs ground truth on a global lat-lon projection for all rollout steps.
    \"\"\"
    num_steps = len(pred_seq)
    fig, axs = plt.subplots(2, num_steps, figsize=(5 * num_steps, 8))
    
    pfull = pred_seq[0].shape[0] // 3
    var_offset = 0 if var == "T" else pfull if var == "U" else 2 * pfull
    ch = var_offset + level_index
    
    nlat, nlon = 180, 360
    lons = np.linspace(0, 360, nlon)
    lats = np.linspace(90, -90, nlat)
    Lon, Lat = np.meshgrid(lons, lats)

    for t in range(num_steps):
        # Regrid True
        true_hpx = true_seq[t][ch]
        true_ll = hpx_to_latlon(true_hpx, level=6, nlat=nlat, nlon=nlon)
        
        # Regrid Pred
        pred_hpx = pred_seq[t][ch]
        pred_ll = hpx_to_latlon(pred_hpx, level=6, nlat=nlat, nlon=nlon)
        
        # Plot True
        im_true = axs[0, t].pcolormesh(Lon, Lat, true_ll, cmap="inferno", shading='auto')
        axs[0, t].set_title(f"Ground Truth {var}, t={t+1}")
        axs[0, t].axis("off")
        
        # Plot Pred
        im_pred = axs[1, t].pcolormesh(Lon, Lat, pred_ll, cmap="inferno", shading='auto')
        axs[1, t].set_title(f"Predicted {var}, t={t+1}")
        axs[1, t].axis("off")

    plt.tight_layout()
    plt.show()

# Run the visualization for the entire Mars globally
plot_global_rollout(pred_seq, true_seq, var="T", level_index=0)
"""

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in new_code.strip().split("\n")]
}
new_cell["source"][-1] = new_cell["source"][-1].strip() # remove trailing newline from last line

nb['cells'].append(new_cell)

with open('mars_weather_dt.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Appended cell to mars_weather_dt.ipynb")
