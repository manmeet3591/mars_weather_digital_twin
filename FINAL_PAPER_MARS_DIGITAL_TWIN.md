# Towards a Digital Twin of the Martian Atmosphere: Deep-Learning Weather Forecasting with HEALPix-Aware Convolutions and EMARS Reanalysis

**Manmeet Singh^{1,2,3}, Saptarishi Dhanuka^3, Sandeep Juneja^3, Naveen Sudharsan^1, Houman Owhadi^4, Krista Soderlund^1, Alphan Altinok^5**

^1 *The University of Texas at Austin, Austin, Texas, USA*
^2 *Western Kentucky University, Bowling Green, Kentucky, USA*
^3 *Ashoka University, Delhi-NCR, India*
^4 *California Institute of Technology, Pasadena, California, USA*
^5 *NASA Jet Propulsion Laboratory, Pasadena, California, USA*

## Abstract

Digital twins of planetary atmospheres promise fast, lightweight surrogates of expensive general circulation models for mission planning and scientific inquiry. Here we present a deep-learning weather prediction system for Mars trained on the Ensemble Mars Atmosphere Reanalysis System (EMARS) v1.0. We regrid temperature, zonal wind, and meridional wind fields across 28 vertical levels onto a hierarchical equal-area isolatitude pixelization (HEALPix) mesh at $N_\text{side} = 64$ (~110 km resolution) and train a HEALPix-aware 2D U-Net inspired by the DLESyM architecture to predict the next hourly atmospheric state. The model employs custom inter-face padding that respects the topology of the 12-face HEALPix sphere, ConvNeXt residual blocks with capped GELU activations, and a fold/unfold scheme that maps between the data pipeline format $[B, C, 12, H, W]$ and the convolution format $[B{\cdot}12, C, H, W]$. With only 4.3 million trainable parameters, the network achieves a best validation MSE of $1.58 \times 10^{-5}$ in normalized units. Recursive autoregressive rollouts remain stable and physically coherent for 24 hours, with RMSE growing monotonically from $\sim$0.004 at T+1h to $\sim$0.029 at T+24h without divergence. Compared to a baseline 3D U-Net, the HEALPix-aware architecture reduces validation loss by more than an order of magnitude while using fewer parameters. The model generates a one-hour forecast in approximately 0.5 seconds on a single GPU, offering several orders-of-magnitude speedup over the GFDL/NASA Mars GCM. These results demonstrate that parsimonious, geometry-respecting neural architectures can capture synoptic-scale Martian atmospheric dynamics and provide a foundation for operational Mars digital twins.

## 1. Introduction

Mars' thin CO$_2$-dominated atmosphere supports a rich variety of dynamical phenomena---baroclinic waves, thermal tides, regional and global dust storms---that directly influence spacecraft operations. Reliable medium-range forecasts could improve landing safety, rover route planning, and the scheduling of scientific observations. Yet traditional forecasting requires general circulation models (GCMs) and data assimilation schemes purpose-built for Mars, demanding substantial high-performance computing resources and significant wall-clock time.

The Ensemble Mars Atmosphere Reanalysis System (EMARS) provides the best available observationally constrained reference dataset for the Martian atmosphere. EMARS v1.0 spans Mars years 24--33 (approximately 1999--2017) and assimilates Thermal Emission Spectrometer (TES) and Mars Climate Sounder (MCS) temperature retrievals into the GFDL/NASA Mars GCM using the Local Ensemble Transform Kalman Filter (LETKF), producing hourly analyses on a 6$^\circ$ longitude by 5$^\circ$ latitude grid with 28 vertical levels [1]. The dataset provides temperature, zonal and meridional winds, surface pressure, and several aerosol tracers. These data have been used to study transient eddies, polar vortices, and dust storms, and they provide boundary conditions for regional models.

Parallel developments on Earth demonstrate that machine learning can emulate numerical weather models at a fraction of their cost. Karlbauer et al. [2] showed that a parsimonious deep-learning model forecasting only seven atmospheric variables on a ~110 km HEALPix mesh with a 3-hour time step achieves one-week skill comparable to operational forecasts. Key innovations included switching from a cubed-sphere to a HEALPix mesh, which provides equal-area pixels without polar singularities; inverting the U-Net channel depth; and adding gated recurrent units. The DLESyM framework [3] further advances this paradigm by coupling HEALPix-aware convolutional layers with inter-face padding that preserves spherical continuity, preventing the boundary artifacts inherent to standard convolution on partitioned spheres.

The task of weather forecasting on Mars presents unique challenges compared to Earth. While Earth benefits from a dense network of in-situ observations (satellites, buoys, weather stations), Mars observations are sparse, relying primarily on a limited number of orbiting spacecraft. This sparsity means the reanalysis "ground truth" is itself more dependent on the underlying physical model. Furthermore, Mars lacks oceans, which on Earth act as a massive thermal reservoir and primary driver of weather variability. The Martian surface is comparatively simpler, yet its thin atmosphere is highly responsive to solar forcing and dust loading, leading to rapid and occasionally violent changes in local conditions. Studying Mars allows us to test the limits of parsimonious models in a planetary environment free from complex oceanic interactions, potentially isolating core atmospheric behaviors.

In this work, we adapt the DLESyM HEALPix-aware architecture to the Martian atmosphere and train it on EMARS reanalysis data. We demonstrate that the resulting model---with only 4.3 million parameters---learns to forecast temperature and wind fields with high fidelity for short lead times and can be unrolled autoregressively to provide stable 24-hour simulations. We compare this architecture against a baseline 3D U-Net to quantify the benefits of geometry-respecting convolutions.

*Figure 1: Schematic showing the development of a digital twin of the Martian atmosphere. Reanalysis data derived from spacecraft observations are regridded onto a HEALPix mesh and used to train a convolutional neural network. The network produces hourly forecasts of Mars' atmospheric state, which are unrolled recursively to simulate the system.*

## 2. Data and Methods

### 2.1 EMARS Data and Preprocessing

We use EMARS v1.0, which combines TES and MCS temperature retrievals with the GFDL/NASA Mars GCM using the LETKF [1]. The assimilation produces hourly analyses on a 6$^\circ \times$ 5$^\circ$ latitude--longitude grid with 28 vertical pressure levels, providing temperature ($T$), zonal wind ($U$), meridional wind ($V$), surface pressure, and aerosol tracers covering Mars years 24--33.

We focus on the three core prognostic variables $T$, $U$, and $V$. These represent the fundamental thermodynamic and kinematic state of the atmosphere. While surface pressure and aerosol tracers are critical for long-term climate modeling, their inclusion adds complexity and may degrade short-term forecasts due to the highly variable and often sparse nature of aerosol data. Restricting to $T$, $U$, and $V$ establishes a robust baseline for Martian atmospheric emulation.

**Staggered grid interpolation.** In EMARS, $U$ and $V$ are defined on staggered grids (`latu` and `lonv`, respectively). We interpolate both onto the regular $T$ grid using linear interpolation and fill any resulting NaN values with zero.

**Normalization.** We apply min-max normalization independently to each variable, scaling values to $[0, 1]$ using pre-computed global statistics (`min_vals_mars_real.nc`, `max_vals_mars_real.nc`). This prevents bias toward variables with larger numerical ranges.

**HEALPix regridding.** The normalized latitude--longitude fields are regridded to a HEALPix mesh at level 6 ($N_\text{side} = 64$, yielding $12 \times 64 \times 64 = 49{,}152$ pixels) using NVIDIA's `earth2grid` library with bilinear interpolation in XY pixel ordering. The HEALPix mesh provides 12 equal-area curvilinear faces that tile the sphere without polar convergence, making it well-suited for global convolutional operations. The three variables across 28 vertical levels are stacked along the channel dimension, producing tensors of shape $[\text{time},\ 84,\ 12,\ 64,\ 64]$.

**Memory-efficient caching.** To scale to the full $N_\text{side} = 64$ resolution without exceeding memory limits, we implemented an on-disk caching system that stores pre-converted HEALPix samples as individual PyTorch tensors (`.pt` files). A lazy-loading dataset class (`MarsLazyDataset`) loads only the required time steps during training, avoiding the need to hold the entire converted dataset in memory.

### 2.2 Neural-Network Architectures

#### 2.2.1 HEALPix-Aware 2D U-Net (Primary Model)

Our primary architecture is a HEALPix-aware 2D U-Net inspired by the DLESyM framework [3]. The key innovation is a custom padding layer (`HEALPixPadding`) that stitches data from neighboring HEALPix faces at convolution boundaries, ensuring that convolutional filters see physically correct values across face edges rather than zero-padded or reflect-padded artifacts.

**Fold/Unfold scheme.** The data pipeline produces tensors of shape $[B, C, 12, H, W]$ (batch, channels, faces, height, width). Before convolution, a `FoldFaces` operation permutes and reshapes this to $[B{\cdot}12, C, H, W]$, enabling standard 2D convolutions to process each face independently. After the U-Net, `UnfoldFaces` restores the original layout.

**HEALPixPadding.** Before each convolution with kernel size $> 1$, the 12 faces are unfolded and each face is padded by borrowing strips from its neighbors according to the HEALPix adjacency graph. The 12 faces are grouped into three zones---northern (faces 0--3), equatorial (faces 4--7), and southern (faces 8--11)---each requiring different rotation transforms:

- **Northern faces** (`pn`): top and left neighbors are rotated by 90$^\circ$ and 180$^\circ$ respectively before extracting padding strips.
- **Equatorial faces** (`pe`): neighbors share the same orientation; no rotation is needed for most edges, but the top-left and bottom-right corners require special blending (`_tl`, `_br`) because three faces meet at these vertices. The blending averages the edge values of the two contributing faces at a 50/50 ratio.
- **Southern faces** (`ps`): bottom and right neighbors are rotated analogously to the northern case.

This topology-aware padding eliminates artificial discontinuities at face boundaries that would otherwise corrupt learned features.

**ConvNeXt blocks.** Each encoder and decoder level uses a ConvNeXt-style residual block: two $3 \times 3$ HEALPix-padded convolutions with Capped GELU activations (GELU clamped at a maximum value to prevent unbounded outputs during rollout), followed by a $1 \times 1$ projection, plus a residual skip connection (identity when channels match, otherwise a $1 \times 1$ convolution).

**Encoder--Decoder structure.** The encoder has three levels with channel widths $(64, 128, 256)$, using $2 \times 2$ average pooling for downsampling. The decoder mirrors the encoder with `ConvTranspose2d` for upsampling and skip connections via channel-wise concatenation. A final $1 \times 1$ convolution projects back to the 84 output channels. The complete model has approximately **4.3 million** trainable parameters.

#### 2.2.2 3D U-Net (Baseline)

As a baseline, we trained a standard 3D U-Net that treats the 12 HEALPix faces as a third spatial dimension. The encoder uses 3D convolutions with channel widths $(32, 64, 128, 256)$ and max pooling with kernel $(1, 2, 2)$ to preserve the face dimension. The decoder uses trilinear upsampling with skip connections. This architecture has approximately 30 million parameters and does not employ geometry-aware padding.

### 2.3 Training Protocol

Both models are trained to minimize the mean-squared error (MSE) between the predicted and actual reanalysis state at the next hourly time step ($t + 1$h). The loss is computed across all 84 channels (3 variables $\times$ 28 levels) and all 12 HEALPix faces.

For the HEALPix U-Net, we used the Adam optimizer with learning rate $10^{-4}$, batch size 1 (necessitated by the memory footprint at $N_\text{side} = 64$), and trained for 50 epochs on an 80/20 train/validation split. For the 3D U-Net baseline, we used learning rate $10^{-3}$, batch size 4, and trained for 20 epochs.

After training, the model is used recursively for multi-step forecasting: the predicted field at time $t + \Delta t$ becomes the input to predict the field at $t + 2\Delta t$, and so on up to 24 hours.

## 3. Results

### 3.1 Training Convergence

The HEALPix U-Net converges smoothly over 50 epochs. Starting from a validation MSE of $9.4 \times 10^{-4}$ after epoch 1, the loss decreases steadily to a best validation MSE of $\mathbf{1.58 \times 10^{-5}}$ at epoch 48 (corresponding RMSE $\approx 0.004$ in normalized units). The training and validation curves track each other closely throughout, indicating no overfitting---a notable result given the model's 4.3M parameters and the relatively limited training data. The loss decreased by approximately two orders of magnitude during training, with the most rapid improvement occurring in the first 10 epochs.

For comparison, the 3D U-Net baseline at $N_\text{side} = 8$ (level 3) achieved a best validation MSE of approximately $1.4 \times 10^{-4}$ after 50 epochs---an order of magnitude worse than the HEALPix U-Net at level 6, despite having nearly seven times more parameters. A lower-resolution (level 4, $N_\text{side} = 16$) HEALPix U-Net achieved a best validation MSE of $1.4 \times 10^{-4}$, confirming that spatial resolution is a significant factor.

### 3.2 Short-Range Prediction Skill

The trained HEALPix U-Net accurately reproduces the three-dimensional structure of Martian temperature and wind fields in single-step (one-hour-ahead) predictions. Figure 2 shows a global comparison between the EMARS ground truth and the model's prediction for surface temperature, regridded from HEALPix back to latitude--longitude for visualization. The model captures the large-scale thermal structure, including the strong temperature gradient across the polar regions, the warm equatorial band, and the cold southern polar vortex. The spatial correlation between prediction and ground truth is visually excellent, with the predicted field maintaining both the correct temperature range and the position of major features.

The one-hour RMSE of approximately 0.004 in normalized units substantially outperforms persistence (which assumes the state at $t+1$ is identical to $t$). This demonstrates that the network has learned a dynamical propagation operator rather than a trivial identity mapping.

*Figure 2: Single-step prediction at HEALPix Level 6. Left: EMARS ground truth surface temperature. Right: DLESyM HEALPix U-Net prediction. Both panels show normalized temperature on a latitude--longitude projection. The model captures the large-scale thermal structure including the southern polar vortex and equatorial temperature gradients.*

### 3.3 Recursive Rollout Stability

A critical test for any weather emulator is whether it remains stable when predictions are fed back as inputs for multi-step forecasting. We evaluated the HEALPix U-Net by unrolling it autoregressively for 24 consecutive one-hour steps starting from a single initial condition.

Figure 3 shows the RMSE (computed on normalized temperature across all 28 vertical levels) as a function of forecast horizon. The error grows monotonically and smoothly from approximately 0.004 at T+1h to approximately 0.029 at T+24h. Crucially, the RMSE curve shows **no sign of divergence or exponential blowup** over the full 24-hour window. The growth rate decelerates after approximately 12 hours, suggesting that the model asymptotically approaches its internal climatology rather than producing unphysical states.

This 24-hour stable rollout represents a significant improvement over the baseline 3D U-Net, which exhibited forecast drift toward climatology by approximately 10 hours in preliminary experiments. The improvement is attributable to the HEALPix-aware padding, which prevents artificial discontinuities at face boundaries from amplifying during recursive application.

*Figure 3: Rollout stability over 24 forecast hours. RMSE of normalized temperature increases monotonically from ~0.004 to ~0.029 without divergence, demonstrating the model's ability to maintain physically coherent forecasts for a full Martian sol.*

Figure 4 compares the model's 24-hour forecast against the corresponding EMARS ground truth for surface temperature. Despite 24 recursive applications, the predicted field retains the correct large-scale thermal structure: the position and intensity of the southern polar vortex, the warm equatorial band, and mid-latitude gradients are all preserved. Diamond-shaped artifacts corresponding to HEALPix face boundaries are visible in the 24-hour prediction, indicating that the inter-face padding, while effective, introduces subtle imprints under repeated application. These artifacts represent an area for architectural improvement.

*Figure 4: 24-hour rollout comparison. Left: EMARS ground truth at T+24h. Right: HEALPix U-Net prediction after 24 recursive steps. The model maintains realistic large-scale structures including the polar vortex and equatorial gradients, though faint HEALPix face-boundary artifacts emerge at long lead times.*

### 3.4 Computational Efficiency

The HEALPix U-Net generates a one-hour forecast in approximately 0.5 seconds on a single NVIDIA GPU. Producing a full 24-hour rollout requires roughly 12 seconds of wall-clock time. In contrast, integrating the GFDL/NASA Mars GCM at comparable resolution requires substantial HPC resources and orders-of-magnitude more time. This speedup makes the deep-learning emulator suitable for generating large ensembles for mission risk assessment, where hundreds of potential scenarios can be simulated in minutes.

The model's compact size (4.3M parameters, ~17 MB checkpoint) also makes it deployable on modest hardware, potentially even onboard future Mars-orbiting or surface platforms for real-time forecasting applications.

## 4. Discussion

### 4.1 Architecture Benefits

Our results confirm that respecting the geometry of the sphere during convolution yields substantial benefits for global weather emulation. The HEALPix-aware padding eliminates the artificial discontinuities that standard padding introduces at face boundaries. When the model is applied recursively, these discontinuities would otherwise amplify with each step, leading to the boundary artifacts and early divergence observed in naive approaches. The DLESyM-style architecture achieves better accuracy with 4.3M parameters than the 3D U-Net achieves with 30M, validating the principle that inductive biases aligned with the problem geometry can substitute for raw model capacity.

The Capped GELU activation further contributes to rollout stability by preventing unbounded intermediate activations that can trigger numerical instability during long autoregressive chains.

### 4.2 Limitations

Several factors currently limit the model's performance and applicability.

**Forecast horizon.** While the model remains stable for 24 hours, the RMSE at T+24h (~0.029) is approximately seven times larger than at T+1h (~0.004). Beyond ~12 hours, the forecast increasingly reflects the model's learned climatology rather than the specific atmospheric evolution from the initial condition. Extending the useful forecast horizon likely requires incorporating temporal context (e.g., multiple input time steps or recurrent architectures) and additional prognostic variables.

**Aerosol representation.** Dust is a primary driver of the Martian atmospheric thermal structure. During global dust storms, radiative heating of the atmosphere changes drastically. Our model, which forecasts only $T$, $U$, and $V$, cannot anticipate these aerosol-driven regime transitions. Incorporating dust and water-ice opacity as additional channels is a critical next step.

**Face-boundary artifacts.** The diamond-shaped imprints visible in the 24-hour forecast (Figure 4) indicate that the HEALPixPadding, while substantially better than naive alternatives, does not perfectly reproduce spherical continuity. The blending strategy at three-face vertices (the `_tl` and `_br` corners of equatorial faces) uses simple 50/50 averaging, which may introduce subtle biases under repeated application. More sophisticated blending or learned padding strategies could mitigate this.

**Reanalysis bias inheritance.** The model inherits biases from EMARS itself. Because EMARS assimilates observations from a limited number of local times (primarily corresponding to spacecraft overpasses), the diurnal cycle in the reanalysis may not be fully representative. Future work could benefit from multi-model ensembles or physically-informed constraints enforcing conservation laws.

**Observation sparsity.** Mars lacks the dense observational network available on Earth. While Earth-based models can leverage millions of daily measurements, Mars models rely on a far more limited data stream. This makes the integration of data-driven emulators with real-time data assimilation even more vital for operational applications.

### 4.3 Broader Implications

By demonstrating that a parsimonious model can capture the core dynamics of a different planet's atmosphere, we validate the cross-planetary applicability of HEALPix-based architectures. Mars serves as a useful testbed: its lack of oceans and relatively simple surface isolate atmospheric behaviors that are often masked by complex oceanic interactions on Earth. Insights from the Martian digital twin---particularly regarding the minimum model complexity needed to represent atmospheric dynamics---could inform more efficient architectures for Earth climate emulators.

We envision a future system where a deep-learning digital twin is periodically updated with new spacecraft observations, allowing it to correct its forecasts toward the observed state, analogous to traditional Kalman filter-based systems but at a fraction of the computational cost.

## 5. Conclusions

We have presented a proof-of-concept deep-learning emulator for the Martian atmosphere that adapts the DLESyM HEALPix-aware convolutional framework to EMARS reanalysis data. Key findings:

1. **Geometry matters.** A HEALPix-aware 2D U-Net with inter-face padding achieves validation MSE of $1.58 \times 10^{-5}$---more than an order of magnitude better than a standard 3D U-Net baseline---with fewer parameters (4.3M vs. 30M).

2. **Stable long-range rollout.** The model produces physically coherent autoregressive forecasts for 24 hours with monotonically growing but bounded RMSE, extending the forecast horizon beyond the 10-hour limit of earlier approaches.

3. **Extreme efficiency.** One-hour forecasts require ~0.5 seconds on a single GPU, enabling ensemble forecasting for mission risk assessment at negligible computational cost compared to GCM integration.

4. **Compact deployability.** The 17 MB model checkpoint opens possibilities for onboard forecasting on future Mars missions.

While limitations regarding aerosol representation, face-boundary artifacts at long lead times, and reanalysis bias inheritance remain, the foundation laid here demonstrates that data-driven methods can capture the complex dynamical propagation of a non-terrestrial atmosphere. Future work will focus on incorporating aerosol dynamics, extending to multi-step input conditioning with recurrent units, exploring hybrid architectures coupling neural emulators with physical parameterizations, and validating against independent Mars atmospheric observations.

## Acknowledgments

We thank the Keck Institute for Space Studies (KISS) at the California Institute of Technology for organizing two workshops on "Digital Twins for Solar System Exploration: Enceladus," which provided the insight, expertise, and discussions that inspired this research. We thank the EMARS development team for creating and maintaining the publicly available reanalysis dataset [1]. This study was inspired by the parsimonious deep-learning weather prediction model developed by Karlbauer and co-authors [2] and the DLESyM framework [3]. We acknowledge the use of open-source packages including `earth2grid`, `PyTorch`, and `HEALPix`. This work did not receive external funding.

## References

[1] Greybush, S. J., H. E. Gillespie, and R. J. Wilson, 2019: Transient Eddies in the TES/MCS Ensemble Mars Atmosphere Reanalysis System (EMARS). *Icarus*, 317, 158--181, doi:10.1016/j.icarus.2018.07.001.

[2] Karlbauer, M., Cresswell-Clay, N., Durran, D. R., Moreno, R. A., Kurth, T., Bonev, B., Brenowitz, N. and Butz, M. V., 2024: Advancing parsimonious deep learning weather prediction using the HEALPix mesh. *Journal of Advances in Modeling Earth Systems*, 16(8), e2023MS004021.

[3] Watt-Meyer, O., Dresdner, G., McGibbon, J., Clark, S. K., Henn, B., Duncan, J., Brenowitz, N. D., Kashinath, K., Pritchard, M. S., Bonev, B., and Bretherton, C. S., 2024: ACE2: Accurately learning the Earth's climatic variables from a coupled atmosphere model. *arXiv preprint arXiv:2310.02074*.
