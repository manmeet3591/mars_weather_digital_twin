import earth2grid
import torch
level = 6
hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
src = earth2grid.latlon.equiangular_lat_lon_grid(32, 64)
z_torch = torch.cos(torch.deg2rad(torch.tensor(src.lat)))
z_torch = z_torch.broadcast_to(src.shape)
regrid = earth2grid.get_regridder(src, hpx)
z_hpx = regrid(z_torch)
nside = 2**level
reshaped = z_hpx.reshape(12, nside, nside)
lat_r = hpx.lat.reshape(12, nside, nside)
lon_r = hpx.lon.reshape(12, nside, nside)
import numpy as np
import torch

import earth2grid

device = "cpu"


# the source grid (from 90N to 90S and 0E to 360E)
ll = earth2grid.latlon.equiangular_lat_lon_grid(721, 1440)

# a 2d grid of target lat lons
target_lat = np.linspace(30, 50, 32)
target_lon = np.linspace(100, 120, 64)
target_lat, target_lon = np.meshgrid(target_lat, target_lon)

# Some source data on the original grid
data = torch.ones([721, 1440]).to(device)

# Create a bilinear regridding object earth2grid
regrid = ll.get_bilinear_regridder_to(target_lat, target_lon)

# need to move the weights to same device and dtype as data
regrid.to(data)

# perform the regridding
out = regrid(data)
assert out.shape == target_lat.shape  # noqa
print("data shape", out.shape)
import torch
import earth2grid

def latlon_to_healpix(data: torch.Tensor, level: int = 6, device: str = "cpu"):
    """
    Regrid 2D lat-lon data to HEALPix format using earth2grid.

    Args:
        data (torch.Tensor): 2D tensor of shape [nlat, nlon].
        level (int): HEALPix resolution level (default: 6 → nside=64).
        device (str): Device to perform computation on ("cpu" or "cuda").

    Returns:
        data_hpx (torch.Tensor): Flat HEALPix data of shape [12 * nside * nside].
        data_faces (torch.Tensor): Reshaped HEALPix data of shape [12, nside, nside].
    """
    # Make sure data is on the correct device
    data = data.to(device)

    # Get source grid matching data shape
    nlat, nlon = data.shape
    src = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)

    # Create HEALPix target grid
    hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
    nside = 2 ** level

    # Create and prepare regridder
    regrid = earth2grid.get_regridder(src, hpx)
    regrid.to(data)

    # Perform regridding
    data_hpx = regrid(data)
    data_faces = data_hpx.reshape(12, nside, nside)

    return data_hpx, data_faces

# Example data: 721 x 1440 grid
data = torch.ones([721, 1440])

# Regrid to HEALPix level 6
data_hpx, data_faces = latlon_to_healpix(data, level=3)

print("Flat HEALPix shape:", data_hpx.shape)       # torch.Size([49152])
print("Face-wise shape:", data_faces.shape)        # torch.Size([12, 64, 64])
import xarray as xr
ds_min = xr.open_dataset('min_vals_mars.nc')
ds_max = xr.open_dataset('max_vals_mars.nc')
import xarray as xr
import glob

# Path to your directory containing the NetCDF files
path = '/content/drive/MyDrive/mars_reanalysis/'

# Get a list of all NetCDF files in the directory
file_list = sorted(glob.glob(path + 'emars_*.nc'))

# # Optionally: create a list of datasets
# datasets = []

for file in ['dummy_mars.nc']:
    ds = xr.open_dataset(file)
    ds_normalized = (ds - ds_min) / (ds_max - ds_min)
ds_normalized
ds_normalized.T
import torch
import xarray as xr
from tqdm import tqdm
import earth2grid

def latlon_to_healpix(data: torch.Tensor, level: int = 6, device: str = "cpu"):
    data = data.to(device)
    nlat, nlon = data.shape
    src = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
    nside = 2 ** level
    regrid = earth2grid.get_regridder(src, hpx)
    regrid.to(data)
    data_faces = regrid(data).reshape(12, nside, nside)
    return data_faces

def convert_to_healpix_array(ds: xr.DataArray, level: int = 6, device: str = "cpu") -> torch.Tensor:
    """
    Convert xarray.DataArray [time, pfull, lat, lon]
    into torch.Tensor [time, pfull, 12, nside, nside].
    """
    time_len = ds.sizes["time"]
    pfull_len = ds.sizes["pfull"]
    nside = 2 ** level

    output = torch.empty((time_len, pfull_len, 12, nside, nside), dtype=torch.float32)

    for t in tqdm(range(time_len), desc="Time"):
        for p in range(pfull_len):
            latlon = torch.tensor(ds.isel(time=t, pfull=p).values, dtype=torch.float32)
            hp = latlon_to_healpix(latlon, level=level, device=device)
            output[t, p] = hp

    return output
healpix_tensor = convert_to_healpix_array(ds_normalized.T, level=3, device="cpu")
print(healpix_tensor.shape)  # torch.Size([time, pfull, 12, 64, 64])
healpix_tensor[:1].shape
# Step 1: Convert each to HEALPix
healpix_T = convert_to_healpix_array(ds_normalized.T, level=3, device="cpu")
healpix_U = convert_to_healpix_array(ds_normalized.U, level=3, device="cpu")
healpix_V = convert_to_healpix_array(ds_normalized.V, level=3, device="cpu")

# Step 2: Concatenate along the pfull dimension (dim=1)
healpix_combined = torch.cat([healpix_T, healpix_U, healpix_V], dim=1)

print("Final shape:", healpix_combined.shape)
# Expected shape: [time, 3 * pfull, 12, 64, 64]
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, up_channels, skip_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(up_channels, up_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv = DoubleConv(up_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        return x

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, trilinear=True, use_ds_conv=False, out_activation=None):
        super(UNet, self).__init__()
        _channels = (32, 64, 128, 256)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c * width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)

        factor = 2 if trilinear else 1

        self.up1 = Up(self.channels[3], self.channels[2], self.channels[2] // factor, trilinear)
        self.up2 = Up(self.channels[2] // factor, self.channels[1], self.channels[1] // factor, trilinear)
        self.up3 = Up(self.channels[1] // factor, self.channels[0], self.channels[0], trilinear)

        self.outc = OutConv(self.channels[0], n_classes, activation=out_activation)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
x = torch.randn(1, 84, 12, 8, 8)
model = UNet(n_channels=84, n_classes=84, out_activation=None)
out = model(x)
print(out.shape)  # Should print: torch.Size([1, 84, 12, 8, 8])
healpix_combined.shape
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
class ncDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).unsqueeze(0)
        y = torch.from_numpy(self.targets[index]).unsqueeze(0)
        # x = self.data[index]
        # y = self.targets[index]
        # x = x.to(dtype=torch.float32)
        # y = y.to(dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.data)
import numpy as np

# Convert to NumPy if not already
healpix_np = healpix_combined.cpu().numpy() if isinstance(healpix_combined, torch.Tensor) else healpix_combined

inputs = healpix_np[:-1]
targets = healpix_np[1:]

train_dataset = ncDataset(inputs, targets)

# Split into train/val
val_split = 0.1
val_size = int(len(train_dataset) * val_split)
train_size = len(train_dataset) - val_size
train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=8, shuffle=False)
def train(model, train_dataloader, val_dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        lr, hr = batch
        lr, hr = lr.to(device), hr.to(device)

        # Remove singleton channel dimension if present
        if lr.dim() == 6 and lr.size(1) == 1:
            lr = lr.squeeze(1)
        if hr.dim() == 6 and hr.size(1) == 1:
            hr = hr.squeeze(1)

        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)

            if lr.dim() == 6 and lr.size(1) == 1:
                lr = lr.squeeze(1)
            if hr.dim() == 6 and hr.size(1) == 1:
                hr = hr.squeeze(1)

            sr = model(lr)
            loss = criterion(sr, hr)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    return train_loss, val_loss

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import wandb

# Initialize WandB
import wandb
wandb.login(key="70f85253c59220a4439123cc3c97280ece560bf5")  # Replace with your API key

wandb.init(project="mars_weather_dt", name="unet3d_run1", config={
    "epochs": 1000,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_fn": "MSELoss",
    "batch_size": train_dataloader.batch_size,
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=84, n_classes=84, out_activation=None).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter()

num_epochs = 1000
print_interval = 10
patience = 50
best_val_loss = float('inf')
counter = 0
best_model = None

is_train = True  # Ensure this is set

if is_train:
    for epoch in range(1, num_epochs + 1):
        train_loss, val_loss = train(model, train_dataloader, val_dataloader, criterion, optimizer, device)

        # Log to TensorBoard
        writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)

        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)
            counter = 0
            # Save best model to wandb
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
        else:
            counter += 1

        if epoch % print_interval == 0:
            print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

    writer.close()
    wandb.finish()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
def predict_sequence(model, input_seq, steps=5):
    """
    Rolls a forecast from input_seq (shape: [T, C, 12, 8, 8])
    for a number of steps, using model's own predictions as inputs.
    """
    preds = []
    current = input_seq.clone()
    for _ in range(steps):
        with torch.no_grad():
            pred = model(current.unsqueeze(0).to(device)).squeeze(0).cpu()
        preds.append(pred)
        current = pred  # roll forward
    return preds

sample_input = torch.from_numpy(inputs[0])  # shape: [84, 12, 8, 8]
sample_target = [torch.from_numpy(targets[i]) for i in range(1, 6)]

forecasted = predict_sequence(model, sample_input, steps=5)
import matplotlib.pyplot as plt

def plot_comparison(preds, targets, level_index=0, variable="T", step=0):
    """
    Plot actual vs predicted for lowest pressure level at given step.
    Assumes shape [84, 12, 8, 8] with T, U, V stacked on pfull.
    """
    # Assuming T is first third of channels
    pfull = preds[step].shape[0] // 3
    t_pred = preds[step][level_index].numpy()
    t_true = targets[step][level_index].numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(t_true[0], cmap="inferno")
    axs[0].set_title(f"Ground Truth T (step={step})")

    axs[1].imshow(t_pred[0], cmap="inferno")
    axs[1].set_title(f"Predicted T (step={step})")

    plt.tight_layout()
    plt.savefig("global_rollout.png")
plot_comparison(forecasted, sample_target, level_index=0, step=0)
def rollout_forecast(model, initial_input, steps=5, device="cpu"):
    """
    Rolls forward predictions using the model's output as next input.

    Args:
        model: trained PyTorch model
        initial_input: torch.Tensor [C, 12, 8, 8] — input at t=0
        steps: number of time steps to forecast
        device: device string

    Returns:
        list of predicted frames: [T1, T2, ..., Tn] each [C, 12, 8, 8]
    """
    model.eval()
    current = initial_input.to(device).unsqueeze(0)  # [1, C, 12, 8, 8]
    predictions = []

    with torch.no_grad():
        for _ in range(steps):
            pred = model(current)  # output: [1, C, 12, 8, 8]
            predictions.append(pred.squeeze(0).cpu())
            current = pred  # feed back prediction as next input

    return predictions
steps = 5
input_0 = torch.from_numpy(inputs[0])  # t=0
true_seq = [torch.from_numpy(targets[i]) for i in range(steps)]  # t=1...t=5

pred_seq = rollout_forecast(model, input_0, steps=steps, device=device)
def plot_rollout_comparison_horizontal(pred_seq, true_seq, var="T", level_index=0, face=0):
    """
    Plot predicted vs ground truth in horizontal layout:
    5 columns (time steps), 2 rows (ground truth & prediction).
    """
    num_steps = len(pred_seq)
    fig, axs = plt.subplots(2, num_steps, figsize=(4 * num_steps, 8))

    pfull = pred_seq[0].shape[0] // 3
    var_offset = 0 if var == "T" else pfull if var == "U" else 2 * pfull
    ch = var_offset + level_index

    for t in range(num_steps):
        pred_face = pred_seq[t][ch][face].numpy()
        true_face = true_seq[t][ch][face].numpy()

        axs[0, t].imshow(true_face, cmap="inferno")
        axs[0, t].set_title(f"GT {var}, t={t+1}", fontsize=12)
        axs[0, t].axis("off")

        axs[1, t].imshow(pred_face, cmap="inferno")
        axs[1, t].set_title(f"Pred {var}, t={t+1}", fontsize=12)
        axs[1, t].axis("off")

    axs[0, 0].set_ylabel("Ground Truth", fontsize=14)
    axs[1, 0].set_ylabel("Prediction", fontsize=14)

    plt.tight_layout()
    plt.savefig("global_rollout.png")
plot_rollout_comparison_horizontal(pred_seq, true_seq, var="T", level_index=0, face=0)

import matplotlib.pyplot as plt
import earth2grid
import torch
import numpy as np

def hpx_to_latlon(data_hpx_faces, level=3, nlat=180, nlon=360, device="cpu"):
    """
    Converts a HEALPix tensor of shape [12, N, N] back to a Lat-Lon grid.
    """
    nside = 2**level
    # Flatten faces to 1D array for earth2grid
    data_flat = data_hpx_faces.reshape(-1).to(device).float()
    
    src_hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
    tgt_ll = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    
    regridder = earth2grid.get_regridder(src_hpx, tgt_ll).to(device).float()
    data_ll = regridder(data_flat)
    return data_ll.reshape(nlat, nlon).cpu().numpy()

def plot_global_rollout(pred_seq, true_seq, var="T", level_index=0):
    """
    Plot predicted vs ground truth on a global lat-lon projection for all rollout steps.
    """
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
        true_ll = hpx_to_latlon(true_hpx, level=3, nlat=nlat, nlon=nlon)
        
        # Regrid Pred
        pred_hpx = pred_seq[t][ch]
        pred_ll = hpx_to_latlon(pred_hpx, level=3, nlat=nlat, nlon=nlon)
        
        # Plot True
        im_true = axs[0, t].pcolormesh(Lon, Lat, true_ll, cmap="inferno", shading='auto')
        axs[0, t].set_title(f"Ground Truth {var}, t={t+1}")
        axs[0, t].axis("off")
        
        # Plot Pred
        im_pred = axs[1, t].pcolormesh(Lon, Lat, pred_ll, cmap="inferno", shading='auto')
        axs[1, t].set_title(f"Predicted {var}, t={t+1}")
        axs[1, t].axis("off")

    plt.tight_layout()
    plt.savefig("global_rollout.png")

# Run the visualization for the entire Mars globally
plot_global_rollout(pred_seq, true_seq, var="T", level_index=0)
