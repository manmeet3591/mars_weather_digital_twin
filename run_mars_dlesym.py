import os
import glob
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import earth2grid
import json
import logging
import gc

# --- CONFIGURATION ---
LEVEL = 6  # N_side = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  
EPOCHS = 50
LR = 1e-4
DATA_PATH = 'mars_data/mars_reanalysis/'
CACHE_DIR = 'mars_cache_hpx_v2'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- MODEL COMPONENTS (DLESyM style) ---

class FoldFaces(nn.Module):
    def forward(self, x):
        B, F, C, H, W = x.shape
        return x.reshape(B * F, C, H, W)

class UnfoldFaces(nn.Module):
    def __init__(self, num_faces=12):
        super().__init__()
        self.num_faces = num_faces
    def forward(self, x):
        NF, C, H, W = x.shape
        return x.reshape(-1, self.num_faces, C, H, W)

class HEALPixPadding(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.p = padding
        self.d = [-2, -1]
        self.fold = FoldFaces()
        self.unfold = UnfoldFaces(num_faces=12)

    def forward(self, data):
        data = self.unfold(data)
        f = [data[:, i] for i in range(12)]
        p00 = self.pn(c=f[0],  t=f[1],  tl=f[2],  l=f[3],  bl=f[3],  b=f[4],  br=f[8],  r=f[5],  tr=f[1])
        p01 = self.pn(c=f[1],  t=f[2],  tl=f[3],  l=f[0],  bl=f[0],  b=f[5],  br=f[9],  r=f[6],  tr=f[2])
        p02 = self.pn(c=f[2],  t=f[3],  tl=f[0],  l=f[1],  bl=f[1],  b=f[6],  br=f[10], r=f[7],  tr=f[3])
        p03 = self.pn(c=f[3],  t=f[0],  tl=f[1],  l=f[2],  bl=f[2],  b=f[7],  br=f[11], r=f[4],  tr=f[0])
        p04 = self.pe(c=f[4],  t=f[0],  tl=self._tl(f[0], f[3]),  l=f[3],  bl=f[7],  b=f[11], br=self._br(f[11], f[8]),  r=f[8],  tr=f[5])
        p05 = self.pe(c=f[5],  t=f[1],  tl=self._tl(f[1], f[0]),  l=f[0],  bl=f[4],  b=f[8],  br=self._br(f[8],  f[9]),  r=f[9],  tr=f[6])
        p06 = self.pe(c=f[6],  t=f[2],  tl=self._tl(f[2], f[1]),  l=f[1],  bl=f[5],  b=f[9],  br=self._br(f[9],  f[10]), r=f[10], tr=f[7])
        p07 = self.pe(c=f[7],  t=f[3],  tl=self._tl(f[3], f[2]),  l=f[2],  bl=f[6],  b=f[10], br=self._br(f[10], f[11]), r=f[11], tr=f[4])
        p08 = self.ps(c=f[8],  t=f[5],  tl=f[0],  l=f[4],  bl=f[11], b=f[11], br=f[10], r=f[9],  tr=f[9])
        p09 = self.ps(c=f[9],  t=f[6],  tl=f[1],  l=f[5],  bl=f[8],  b=f[8],  br=f[11], r=f[10], tr=f[10])
        p10 = self.ps(c=f[10], t=f[7],  tl=f[2],  l=f[6],  bl=f[9],  b=f[9],  br=f[8],  r=f[11], tr=f[11])
        p11 = self.ps(c=f[11], t=f[4],  tl=f[3],  l=f[7],  bl=f[10], b=f[10], br=f[9],  r=f[8],  tr=f[8])
        return self.fold(torch.stack([p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11], dim=1))

    def pn(self, c, t, tl, l, bl, b, br, r, tr):
        p, d = self.p, self.d
        c = torch.cat((t.rot90(1, d)[..., -p:, :], c, b[..., :p, :]), dim=-2)
        left = torch.cat((tl.rot90(2, d)[..., -p:, -p:], l.rot90(-1, d)[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = torch.cat((tr[..., -p:, :p], r[..., :p], br[..., :p, :p]), dim=-2)
        return torch.cat((left, c, right), dim=-1)

    def pe(self, c, t, tl, l, bl, b, br, r, tr):
        p = self.p
        c = torch.cat((t[..., -p:, :], c, b[..., :p, :]), dim=-2)
        left = torch.cat((tl[..., -p:, -p:], l[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = torch.cat((tr[..., -p:, :p], r[..., :p], br[..., :p, :p]), dim=-2)
        return torch.cat((left, c, right), dim=-1)

    def ps(self, c, t, tl, l, bl, b, br, r, tr):
        p, d = self.p, self.d
        c = torch.cat((t[..., -p:, :], c, b.rot90(1, d)[..., :p, :]), dim=-2)
        left = torch.cat((tl[..., -p:, -p:], l[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = torch.cat((tr[..., -p:, :p], r.rot90(-1, d)[..., :p], br.rot90(2, d)[..., :p, :p]), dim=-2)
        return torch.cat((left, c, right), dim=-1)

    def _tl(self, t, l):
        ret = torch.zeros_like(t)[..., :self.p, :self.p]
        ret[..., -1, -1] = 0.5 * t[..., -1, 0] + 0.5 * l[..., 0, -1]
        for i in range(1, self.p):
            ret[..., -i-1, -i:] = t[..., -i-1, :i]
            ret[..., -i:, -i-1] = l[..., :i, -i-1]
            ret[..., -i-1, -i-1] = 0.5 * t[..., -i-1, 0] + 0.5 * l[..., 0, -i-1]
        return ret

    def _br(self, b, r):
        ret = torch.zeros_like(b)[..., :self.p, :self.p]
        ret[..., 0, 0] = 0.5 * b[..., 0, -1] + 0.5 * r[..., -1, 0]
        for i in range(1, self.p):
            ret[..., :i, i] = r[..., -i:, i]
            ret[..., i, :i] = b[..., i, -i:]
            ret[..., i, i] = 0.5 * b[..., i, -1] + 0.5 * r[..., -1, i]
        return ret

class HEALPixConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        layers = []
        if kernel_size > 1:
            pad_size = ((kernel_size - 1) // 2) * dilation
            layers.append(HEALPixPadding(padding=pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        layers.append(self.conv)
        self.layers = nn.Sequential(*layers)
    def forward(self, x): return self.layers(x)

class CappedGELU(nn.Module):
    def __init__(self, cap_value=10.0):
        super().__init__()
        self.gelu = nn.GELU()
        self.cap = cap_value
    def forward(self, x): return torch.clamp(self.gelu(x), max=self.cap)

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels=None):
        super().__init__()
        if latent_channels is None: latent_channels = max(in_channels, out_channels)
        if in_channels == out_channels: self.skip = nn.Identity()
        else: self.skip = HEALPixConv2d(in_channels, out_channels, kernel_size=1)
        self.block = nn.Sequential(
            HEALPixConv2d(in_channels, latent_channels, kernel_size=3),
            CappedGELU(),
            HEALPixConv2d(latent_channels, latent_channels, kernel_size=3),
            CappedGELU(),
            HEALPixConv2d(latent_channels, out_channels, kernel_size=1),
        )
    def forward(self, x): return self.skip(x) + self.block(x)

class HEALPixEncoder(nn.Module):
    def __init__(self, input_channels, n_channels=(64, 128, 256)):
        super().__init__()
        self.levels = nn.ModuleList()
        in_ch = input_channels
        for i, out_ch in enumerate(n_channels):
            level = nn.Sequential()
            if i > 0: level.add_module("pool", nn.AvgPool2d(kernel_size=2))
            level.add_module("conv", ConvNeXtBlock(in_ch, out_ch))
            self.levels.append(level)
            in_ch = out_ch
    def forward(self, x):
        outputs = []
        for level in self.levels:
            x = level(x)
            outputs.append(x)
        return outputs

class HEALPixDecoder(nn.Module):
    def __init__(self, n_channels=(256, 128, 64), output_channels=84):
        super().__init__()
        self.levels = nn.ModuleList()
        for i in range(len(n_channels)):
            level = nn.ModuleDict()
            if i == 0:
                level["upsamp"] = None
                level["conv"] = ConvNeXtBlock(n_channels[0], n_channels[0])
            else:
                level["upsamp"] = nn.ConvTranspose2d(n_channels[i - 1], n_channels[i], kernel_size=2, stride=2)
                level["conv"] = ConvNeXtBlock(n_channels[i] * 2, n_channels[i])
            self.levels.append(level)
        self.output_layer = HEALPixConv2d(n_channels[-1], output_channels, kernel_size=1)
    def forward(self, encodings):
        x = encodings[-1]
        for i, level in enumerate(self.levels):
            if level["upsamp"] is not None:
                x = level["upsamp"](x)
                skip = encodings[-(i + 1)]
                x = torch.cat([x, skip], dim=1)
            x = level["conv"](x)
        return self.output_layer(x)

class HEALPixUNet(nn.Module):
    def __init__(self, input_channels, output_channels, n_channels=(64, 128, 256)):
        super().__init__()
        self.fold = FoldFaces()
        self.unfold = UnfoldFaces(num_faces=12)
        self.encoder = HEALPixEncoder(input_channels, n_channels)
        self.decoder = HEALPixDecoder(n_channels[::-1], output_channels)
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4) # [B, 12, C, H, W]
        x = self.fold(x)
        encodings = self.encoder(x)
        output = self.decoder(encodings)
        output = self.unfold(output)
        output = output.permute(0, 2, 1, 3, 4) # [B, C, 12, H, W]
        return output

# --- DATA LOADING (Strictly Memory Efficient) ---

def cache_healpix_per_sample(file_list, ds_min, ds_max, level=6):
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_sample_files = []
    
    nlat, nlon = 0, 0
    src, hpx, regrid = None, None, None

    for i, file_path in enumerate(file_list):
        log.info(f"Processing source file: {file_path}")
        ds = xr.open_dataset(file_path)
        ds_norm = (ds - ds_min) / (ds_max - ds_min)
        
        # Grid interp
        u_interp = ds_norm['U'].interp(latu=ds_norm.lat)
        ds_norm['U'] = u_interp.drop_vars('latu').fillna(0.0)
        v_interp = ds_norm['V'].interp(lonv=ds_norm.lon)
        ds_norm['V'] = v_interp.drop_vars('lonv').fillna(0.0)
        ds_norm['T'] = ds_norm['T'].fillna(0.0)
        
        if regrid is None:
            nlat, nlon = ds.sizes["lat"], ds.sizes["lon"]
            src = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
            hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
            regrid = earth2grid.get_regridder(src, hpx).to("cpu").float()

        time_len = ds.sizes["time"]
        pfull_len = ds.sizes["pfull"]
        nside = 2 ** level
        
        for t in range(time_len):
            cache_name = os.path.join(CACHE_DIR, f"sample_f{i}_t{t:04d}.pt")
            all_sample_files.append(cache_name)
            
            if os.path.exists(cache_name):
                continue
                
            # Process one time step across all vertical levels and variables
            # [84, lat, lon]
            t_data = []
            for var in ['T', 'U', 'V']:
                v = torch.from_numpy(ds_norm[var].isel(time=t).values).float() # [28, lat, lon]
                h = regrid(v.reshape(-1, nlat, nlon)) # [28, 12*N*N]
                t_data.append(h.reshape(pfull_len, 12, nside, nside))
            
            combined = torch.cat(t_data, dim=0) # [84, 12, N, N]
            if torch.isnan(combined).any(): combined = torch.nan_to_num(combined, 0.0)
            
            torch.save(combined, cache_name)
            if t % 100 == 0: log.info(f"  Cached time step {t}/{time_len}")
        
        del ds, ds_norm
        gc.collect()
        
    return all_sample_files

class MarsLazyDataset(Dataset):
    def __init__(self, sample_files):
        self.files = sample_files

    def __len__(self):
        return len(self.files) - 1

    def __getitem__(self, idx):
        # Strictly lazy: load only when needed
        x = torch.load(self.files[idx], map_location="cpu")
        y = torch.load(self.files[idx+1], map_location="cpu")
        return x, y

# --- MAIN EXECUTION ---
log.info("Loading config and metadata...")
file_list = sorted(glob.glob(DATA_PATH + '*.nc'))[:2]
ds_min = xr.open_dataset('min_vals_mars_real.nc')
ds_max = xr.open_dataset('max_vals_mars_real.nc')

log.info("Checking/Creating HEALPix Sample Cache (Lazy)...")
sample_files = cache_healpix_per_sample(file_list, ds_min, ds_max, level=LEVEL)

log.info(f"Initializing Lazy Dataset with {len(sample_files)} samples...")
dataset = MarsLazyDataset(sample_files)
train_size = int(0.8 * len(dataset))
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2)

# --- TRAINING ---
model = HEALPixUNet(84, 84).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

progress = []
progress_file = f"progress_dlesym_level{LEVEL}.json"
log.info("Starting training...")
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    log.info(f"Epoch {epoch+1}/{EPOCHS} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    progress.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
    with open(progress_file, "w") as f: json.dump(progress, f, indent=2)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"best_model_dlesym_level{LEVEL}.pth")
        log.info(f"  Saved best model! Val: {best_val_loss:.6f}")

log.info("Training complete.")
