        self.up3 = Up(channels[1] // factor, channels[0], channels[0], trilinear)
        self.outc = OutConv(channels[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# HEALPix-aware 2D UNet (DLESyM-style)
# ---------------------------------------------------------------------------


class FoldFaces(nn.Module):
    """[B, 12, C, H, W] -> [B*12, C, H, W]"""
    def forward(self, x):
        B, F, C, H, W = x.shape
        return x.reshape(B * F, C, H, W)


class UnfoldFaces(nn.Module):
    """[B*12, C, H, W] -> [B, 12, C, H, W]"""
    def __init__(self, num_faces=12):
        super().__init__()
        self.num_faces = num_faces

    def forward(self, x):
        NF, C, H, W = x.shape
        return x.reshape(-1, self.num_faces, C, H, W)


class HEALPixPadding(nn.Module):
    """
    Pads each of the 12 HEALPix faces using data from neighboring faces.
    Expects face-folded input [B*12, C, H, W]. Internally unfolds, pads, re-folds.

    Face layout:
      Northern: 0, 1, 2, 3
      Equatorial: 4, 5, 6, 7
      Southern: 8, 9, 10, 11
    """

    def __init__(self, padding):
        super().__init__()
        if not isinstance(padding, int) or padding < 1:
            raise ValueError(f"padding must be int >= 1, got {padding}")
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

        return self.fold(torch.stack(
            [p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11], dim=1))

    def pn(self, c, t, tl, l, bl, b, br, r, tr):
        """Pad a northern hemisphere face."""
        p, d = self.p, self.d
        c = torch.cat((t.rot90(1, d)[..., -p:, :], c, b[..., :p, :]), dim=-2)
        left = torch.cat((tl.rot90(2, d)[..., -p:, -p:], l.rot90(-1, d)[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = torch.cat((tr[..., -p:, :p], r[..., :p], br[..., :p, :p]), dim=-2)
        return torch.cat((left, c, right), dim=-1)

    def pe(self, c, t, tl, l, bl, b, br, r, tr):
        """Pad an equatorial face."""
        p = self.p
        c = torch.cat((t[..., -p:, :], c, b[..., :p, :]), dim=-2)
        left = torch.cat((tl[..., -p:, -p:], l[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = torch.cat((tr[..., -p:, :p], r[..., :p], br[..., :p, :p]), dim=-2)
        return torch.cat((left, c, right), dim=-1)

    def ps(self, c, t, tl, l, bl, b, br, r, tr):
        """Pad a southern hemisphere face."""
        p, d = self.p, self.d
        c = torch.cat((t[..., -p:, :], c, b.rot90(1, d)[..., :p, :]), dim=-2)
        left = torch.cat((tl[..., -p:, -p:], l[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = torch.cat((tr[..., -p:, :p], r.rot90(-1, d)[..., :p], br.rot90(2, d)[..., :p, :p]), dim=-2)
        return torch.cat((left, c, right), dim=-1)

    def _tl(self, t, l):
        """Assemble undefined top-left corner for equatorial faces (50% blend)."""
        ret = torch.zeros_like(t)[..., :self.p, :self.p]
        ret[..., -1, -1] = 0.5 * t[..., -1, 0] + 0.5 * l[..., 0, -1]
        for i in range(1, self.p):
            ret[..., -i-1, -i:] = t[..., -i-1, :i]
            ret[..., -i:, -i-1] = l[..., :i, -i-1]
            ret[..., -i-1, -i-1] = 0.5 * t[..., -i-1, 0] + 0.5 * l[..., 0, -i-1]
        return ret

    def _br(self, b, r):
        """Assemble undefined bottom-right corner for equatorial faces (50% blend)."""
        ret = torch.zeros_like(b)[..., :self.p, :self.p]
        ret[..., 0, 0] = 0.5 * b[..., 0, -1] + 0.5 * r[..., -1, 0]
        for i in range(1, self.p):
            ret[..., :i, i] = r[..., -i:, i]
            ret[..., i, :i] = b[..., i, -i:]
            ret[..., i, i] = 0.5 * b[..., i, -1] + 0.5 * r[..., -1, i]
        return ret


class HEALPixConv2d(nn.Module):
    """Conv2d with automatic HEALPixPadding for kernel_size > 1."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        layers = []
        if kernel_size > 1:
            pad_size = ((kernel_size - 1) // 2) * dilation
            layers.append(HEALPixPadding(padding=pad_size))
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=0 if kernel_size > 1 else 0,
            dilation=dilation, groups=groups, bias=bias)
        layers.append(self.conv)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CappedGELU(nn.Module):
    """GELU activation clamped at a max value to prevent unbounded outputs."""

    def __init__(self, cap_value=1.0):
        super().__init__()
        self.gelu = nn.GELU()
        self.cap = cap_value

    def forward(self, x):
        return torch.clamp(self.gelu(x), max=self.cap)


class ConvNeXtBlock(nn.Module):
    """
    DLESyM-style ConvNeXt block with HEALPix-aware padding.
    3×3 conv → CappedGELU → 3×3 conv → CappedGELU → 1×1 conv + residual
    """

    def __init__(self, in_channels, out_channels, latent_channels=None, upscale_factor=4):
        super().__init__()
        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = HEALPixConv2d(in_channels, out_channels, kernel_size=1)

        self.block = nn.Sequential(
            HEALPixConv2d(in_channels, latent_channels * upscale_factor, kernel_size=3),
            CappedGELU(),
            HEALPixConv2d(latent_channels * upscale_factor, latent_channels * upscale_factor, kernel_size=3),
            CappedGELU(),
            HEALPixConv2d(latent_channels * upscale_factor, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.skip(x) + self.block(x)


class HEALPixEncoder(nn.Module):
    """UNet encoder: progressive downsampling with ConvNeXt blocks."""

    def __init__(self, input_channels, n_channels=(64, 128, 256)):
        super().__init__()
        self.n_channels = n_channels
        self.levels = nn.ModuleList()
        in_ch = input_channels
        for i, out_ch in enumerate(n_channels):
            level = nn.Sequential()
            if i > 0:
                level.add_module("pool", nn.AvgPool2d(kernel_size=2))
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
    """UNet decoder: progressive upsampling with skip connections."""

    def __init__(self, n_channels=(256, 128, 64), output_channels=14):
        super().__init__()
        self.n_channels = n_channels
        self.levels = nn.ModuleList()

        for i in range(len(n_channels)):
            level = nn.ModuleDict()
            if i == 0:
                level["upsamp"] = None
                level["conv"] = ConvNeXtBlock(n_channels[0], n_channels[0])
            else:
                level["upsamp"] = nn.ConvTranspose2d(
                    n_channels[i - 1], n_channels[i],
                    kernel_size=2, stride=2)
                level["conv"] = ConvNeXtBlock(
                    n_channels[i] * 2, n_channels[i])
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
    """
    Full DLESyM-style HEALPix UNet.
    Input/output: [B, C, 12, H, W] (our data pipeline format).
    Internally uses [B*12, C, H, W] with HEALPix-aware padding.
    """

    def __init__(self, input_channels, output_channels, n_channels=(64, 128, 256)):
        super().__init__()
        self.fold = FoldFaces()
        self.unfold = UnfoldFaces(num_faces=12)
        self.encoder = HEALPixEncoder(input_channels, n_channels)
        self.decoder = HEALPixDecoder(n_channels[::-1], output_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.fold(x)
        encodings = self.encoder(x)
        output = self.decoder(encodings)
        output = self.unfold(output)
        output = output.permute(0, 2, 1, 3, 4)
        return output


def build_deterministic_model(device, arch="healpix", width_multiplier=1, n_channels=(64, 128, 256)):
    if arch == "healpix":
        model = HEALPixUNet(
            input_channels=COND_CHANNELS,
            output_channels=TARGET_CHANNELS,
            n_channels=n_channels,
        ).to(device)
        nparams = sum(p.numel() for p in model.parameters())
        log.info(f"HEALPixUNet parameters: {nparams:,}  n_channels={n_channels}")
    else:
        model = UNet3D(
            n_channels=COND_CHANNELS,
            n_classes=TARGET_CHANNELS,
            width_multiplier=width_multiplier,
            trilinear=True,
        ).to(device)
        nparams = sum(p.numel() for p in model.parameters())
        log.info(f"UNet3D parameters: {nparams:,}  (width_multiplier={width_multiplier})")
    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

