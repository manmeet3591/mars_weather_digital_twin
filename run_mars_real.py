import os
import glob
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import earth2grid
import wandb

# --- CONFIGURATION ---
LEVEL = 3  # For proof of concept
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-3
DATA_PATH = 'mars_data/mars_reanalysis/'

# --- UTILITIES ---
def latlon_to_healpix(data: torch.Tensor, level: int = 6, device: str = "cpu"):
    data = data.to(device)
    nlat, nlon = data.shape
    src = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
    nside = 2 ** level
    regrid = earth2grid.get_regridder(src, hpx).to(device).float()
    data_faces = regrid(data.float()).reshape(12, nside, nside)
    return data_faces

def convert_to_healpix_array(ds: xr.DataArray, level: int = 6, device: str = "cpu") -> torch.Tensor:
    time_len = ds.sizes["time"]
    pfull_len = ds.sizes["pfull"]
    nside = 2 ** level
    output = torch.empty((time_len, pfull_len, 12, nside, nside), dtype=torch.float32)
    for t in tqdm(range(time_len), desc="Converting"):
        for p in range(pfull_len):
            latlon = torch.tensor(ds.isel(time=t, pfull=p).values, dtype=torch.float32)
            hp = latlon_to_healpix(latlon, level=level, device=device)
            output[t, p] = hp
    return output

# --- MODEL ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear')
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.up1 = Up(256 + 128, 64)
        self.up2 = Up(64 + 64, 32)
        self.up3 = Up(32 + 32, 32)
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

# --- DATA LOADING ---
print("Loading real data and normalizing...")
file_list = sorted(glob.glob(DATA_PATH + '*.nc'))[:2] # use 2 files to keep it manageable
ds = xr.open_mfdataset(file_list, combine='nested', concat_dim='time')
ds_min = xr.open_dataset('min_vals_mars_real.nc')
ds_max = xr.open_dataset('max_vals_mars_real.nc')
ds_normalized = (ds - ds_min) / (ds_max - ds_min)

# Staggered grid fix: interpolate U and V to T's grid
print("Interpolating U and V to T grid...")
u_interp = ds_normalized['U'].interp(latu=ds_normalized.lat)
ds_normalized['U'] = u_interp.drop_vars('latu').fillna(0.0)

v_interp = ds_normalized['V'].interp(lonv=ds_normalized.lon)
ds_normalized['V'] = v_interp.drop_vars('lonv').fillna(0.0)

# Fill any NaNs in T as well (just in case)
ds_normalized['T'] = ds_normalized['T'].fillna(0.0)

print("Converting to HEALPix...")
hpx_T = convert_to_healpix_array(ds_normalized.T, level=LEVEL, device="cpu")
hpx_U = convert_to_healpix_array(ds_normalized.U, level=LEVEL, device="cpu")
hpx_V = convert_to_healpix_array(ds_normalized.V, level=LEVEL, device="cpu")
hpx_data = torch.cat([hpx_T, hpx_U, hpx_V], dim=1) # [T, 3*28, 12, N, N]

# Check for NaNs in final tensor
if torch.isnan(hpx_data).any():
    print("Warning: NaNs found in hpx_data, filling with 0")
    hpx_data = torch.nan_to_num(hpx_data, 0.0)

class MarsDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data) - 1
    def __getitem__(self, idx): return self.data[idx], self.data[idx+1]

dataset = MarsDataset(hpx_data)
train_size = int(0.8 * len(dataset))
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# --- TRAINING ---
model = UNet(84, 84).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

wandb.init(project="mars_weather_dt_real", name="real_data_run")

print("Starting training...")
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch+1})
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model_real.pth")

# --- VISUALIZATION ---
def hpx_to_latlon(data_hpx_faces, level=LEVEL, nlat=180, nlon=360, device="cpu"):
    nside = 2**level
    data_flat = data_hpx_faces.reshape(-1).to(device).float()
    src_hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
    tgt_ll = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    regridder = earth2grid.get_regridder(src_hpx, tgt_ll).to(device).float()
    data_ll = regridder(data_flat)
    return data_ll.reshape(nlat, nlon).cpu().numpy()

def plot_global_comparison(model, dataset, step=0, var="T", level_index=0):
    model.eval()
    sample_input, sample_target = dataset[step]
    with torch.no_grad():
        pred = model(sample_input.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
    
    ch = level_index if var == "T" else 28 + level_index if var == "U" else 56 + level_index
    true_ll = hpx_to_latlon(sample_target[ch])
    pred_ll = hpx_to_latlon(pred[ch])
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    im0 = axs[0].imshow(true_ll, cmap="inferno", extent=[0, 360, -90, 90])
    axs[0].set_title(f"Ground Truth {var} (Level {level_index})")
    plt.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(pred_ll, cmap="inferno", extent=[0, 360, -90, 90])
    axs[1].set_title(f"Predicted {var} (Level {level_index})")
    plt.colorbar(im1, ax=axs[1])
    
    plt.tight_layout()
    plt.savefig("real_data_comparison.png")
    print("Comparison plot saved as real_data_comparison.png")

plot_global_comparison(model, dataset, var="T", level_index=0)
wandb.finish()
