# Towards a Digital Twin of the Martian Atmosphere: Deep-Learning Weather Forecasting with HEALPix-Aware Convolutions and EMARS Reanalysis

**Manmeet Singh^{1,2,3}, Saptarishi Dhanuka^3, Sandeep Juneja^3, Naveen Sudharsan^1, Houman Owhadi^4, Krista Soderlund^1, Alphan Altinok^5**

^1 *The University of Texas at Austin, Austin, Texas, USA*
^2 *Western Kentucky University, Bowling Green, Kentucky, USA*
^3 *Ashoka University, Delhi-NCR, India*
^4 *California Institute of Technology, Pasadena, California, USA*
^5 *NASA Jet Propulsion Laboratory, Pasadena, California, USA*

## Abstract

Digital twins of planetary atmospheres promise fast, lightweight surrogates of expensive general circulation models for mission planning and scientific inquiry. Here we present a deep-learning weather prediction system for Mars trained on the Ensemble Mars Atmosphere Reanalysis System (EMARS) v1.0. We regrid temperature, zonal wind, and meridional wind fields across 28 vertical levels onto a hierarchical equal-area isolatitude pixelization (HEALPix) mesh at N_side = 64 (~110 km resolution) and train a HEALPix-aware 2D U-Net inspired by the DLESyM architecture [1] to predict the next hourly atmospheric state. The model employs custom inter-face padding that respects the topology of the 12-face HEALPix sphere, ConvNeXt residual blocks with capped GELU activations [2, 3], and a fold/unfold scheme that maps between the data pipeline format [B, C, 12, H, W] and the convolution format [B*12, C, H, W]. With only 4.3 million trainable parameters, the network achieves a best validation MSE of 1.58 x 10^-5 in normalized units. Recursive autoregressive rollouts remain stable and physically coherent for 24 hours, with RMSE growing monotonically from ~0.004 at T+1h to ~0.029 at T+24h without divergence. Compared to a baseline 3D U-Net [4, 5], the HEALPix-aware architecture reduces validation loss by more than an order of magnitude while using fewer parameters. The model generates a one-hour forecast in approximately 0.5 seconds on a single GPU, offering several orders-of-magnitude speedup over the GFDL/NASA Mars GCM. These results demonstrate that parsimonious, geometry-respecting neural architectures can capture synoptic-scale Martian atmospheric dynamics and provide a foundation for operational Mars digital twins.

## 1. Introduction

Mars' thin CO2-dominated atmosphere supports a rich variety of dynamical phenomena---baroclinic waves, thermal tides, regional and global dust storms---that directly influence spacecraft operations [6]. Reliable medium-range forecasts could improve landing safety, rover route planning, and the scheduling of scientific observations. Yet traditional forecasting requires general circulation models (GCMs) and data assimilation schemes purpose-built for Mars, demanding substantial high-performance computing resources and significant wall-clock time [7].

The Ensemble Mars Atmosphere Reanalysis System (EMARS) provides the best available observationally constrained reference dataset for the Martian atmosphere. EMARS v1.0 spans Mars years 24--33 (approximately 1999--2017) and assimilates Thermal Emission Spectrometer (TES) and Mars Climate Sounder (MCS) temperature retrievals into the GFDL/NASA Mars GCM using the Local Ensemble Transform Kalman Filter (LETKF), producing hourly analyses on a 6 deg longitude by 5 deg latitude grid with 28 vertical levels [8]. The dataset provides temperature, zonal and meridional winds, surface pressure, and several aerosol tracers. These data have been used to study transient eddies, polar vortices, and dust storms [9].

Parallel developments on Earth demonstrate that machine learning can emulate numerical weather models at a fraction of their cost. Karlbauer et al. [10] showed that a parsimonious deep-learning model forecasting only seven atmospheric variables on a ~110 km HEALPix mesh [11] with a 3-hour time step achieves one-week skill comparable to operational forecasts. Key innovations included switching from a cubed-sphere to a HEALPix mesh, which provides equal-area pixels without polar singularities; inverting the U-Net channel depth; and adding gated recurrent units. The DLESyM framework [1] further advances this paradigm by coupling HEALPix-aware convolutional layers with inter-face padding that preserves spherical continuity. These efforts build on a broader wave of AI weather models including Pangu-Weather [12], GraphCast [13], and FourCastNet [14], which have demonstrated competitive skill with operational numerical weather prediction systems on Earth.

Despite this progress on Earth, comparatively little work has applied deep learning to Mars atmospheric prediction. Existing efforts have focused on dust storm detection and monitoring from orbital imagery [20, 21], and on conventional GCM-based forecasting for mission support [22]. The hybrid NeuralGCM framework [23], which couples a differentiable dynamical core with learned physics parameterizations, represents a promising paradigm for planets where dust and radiation dominate the energy budget; however, it has not yet been applied to Mars.

Mars differs fundamentally from Earth in ways that shape the weather prediction problem. Mars has roughly half Earth's radius (3,390 km vs. 6,371 km) and 38% of its surface gravity. Its atmosphere is ~95% CO2 with a mean surface pressure of only ~610 Pa---less than 1% of Earth's 101,325 Pa [7]. Surface temperatures range from ~140 K at the winter poles to ~300 K at equatorial noon, a span of ~160 K compared to Earth's ~100 K range, and diurnal swings of 60--80 K are common [6]. Wind speeds can reach 30 m/s, comparable to Earth's surface winds, but the thin atmosphere carries far less momentum. Crucially, Mars lacks oceans, which on Earth provide massive thermal inertia and drive weather variability through evaporation and latent heat release. The Martian surface responds almost instantaneously to solar forcing, producing rapid thermal tides and abrupt dust-driven regime transitions.

These differences make Martian weather prediction both easier and harder than on Earth. The simpler surface boundary (no oceans, no vegetation, minimal topographic moisture effects) reduces the number of coupled processes a model must represent. However, the extreme sparsity of observations---limited to a handful of orbiting spacecraft and surface rovers [6]---means the reanalysis "ground truth" is more dependent on the underlying physical model. Furthermore, dust storms can restructure the atmospheric thermal profile within hours [15], introducing abrupt nonlinearities absent from Earth's weather. Mars thus provides a challenging yet instructive testbed for data-driven weather models: success here in a data-sparse, dust-dominated regime would validate the broader applicability of these architectures beyond Earth.

In this work, we adapt the DLESyM HEALPix-aware architecture to the Martian atmosphere and train it on EMARS reanalysis data. We demonstrate that the resulting model---with only 4.3 million parameters---learns to forecast temperature and wind fields with high fidelity for short lead times and can be unrolled autoregressively to provide stable 24-hour simulations. We compare this architecture against a baseline 3D U-Net to quantify the benefits of geometry-respecting convolutions.

## 2. Data and Methods

### 2.1 EMARS Data and Preprocessing

We use EMARS v1.0, which combines TES and MCS temperature retrievals with the GFDL/NASA Mars GCM using the LETKF [8, 9]. The assimilation produces hourly analyses on a 6 deg x 5 deg latitude--longitude grid with 28 vertical pressure levels, providing temperature (T), zonal wind (U), meridional wind (V), surface pressure, and aerosol tracers covering Mars years 24--33.

We focus on the three core prognostic variables T, U, and V. These represent the fundamental thermodynamic and kinematic state of the atmosphere. While surface pressure and aerosol tracers are critical for long-term climate modeling [15, 16], their inclusion adds complexity and may degrade short-term forecasts due to the highly variable and often sparse nature of aerosol data. Restricting to T, U, and V establishes a robust baseline for Martian atmospheric emulation.

**Staggered grid interpolation.** In EMARS, U and V are defined on staggered grids (`latu` and `lonv`, respectively). We interpolate both onto the regular T grid using linear interpolation and fill any resulting NaN values with zero.

**Normalization.** We apply min-max normalization independently to each variable, scaling values to [0, 1] using pre-computed global statistics. This prevents bias toward variables with larger numerical ranges.

**HEALPix regridding.** The normalized latitude--longitude fields are regridded to a HEALPix mesh at level 6 (N_side = 64, yielding 12 x 64 x 64 = 49,152 pixels) using NVIDIA's `earth2grid` library with bilinear interpolation in XY pixel ordering [17]. The HEALPix mesh [11] provides 12 equal-area curvilinear faces that tile the sphere without polar convergence, making it well-suited for global convolutional operations. The three variables across 28 vertical levels are stacked along the channel dimension, producing tensors of shape [time, 84, 12, 64, 64].

**Memory-efficient caching.** To scale to the full N_side = 64 resolution without exceeding memory limits, we implemented an on-disk caching system that stores pre-converted HEALPix samples as individual PyTorch [18] tensors. A lazy-loading dataset class loads only the required time steps during training, avoiding the need to hold the entire converted dataset in memory.

### 2.2 Neural-Network Architectures

#### 2.2.1 HEALPix-Aware 2D U-Net (Primary Model)

Our primary architecture is a HEALPix-aware 2D U-Net inspired by the DLESyM framework [1]. The key innovation is a custom padding layer (HEALPixPadding) that stitches data from neighboring HEALPix faces at convolution boundaries, ensuring that convolutional filters see physically correct values across face edges rather than zero-padded or reflect-padded artifacts.

**Fold/Unfold scheme.** The data pipeline produces tensors of shape [B, C, 12, H, W] (batch, channels, faces, height, width). Before convolution, a FoldFaces operation permutes and reshapes this to [B*12, C, H, W], enabling standard 2D convolutions to process each face independently. After the U-Net, UnfoldFaces restores the original layout.

**HEALPixPadding.** Before each convolution with kernel size > 1, the 12 faces are unfolded and each face is padded by borrowing strips from its neighbors according to the HEALPix adjacency graph. The 12 faces are grouped into three zones---northern (faces 0--3), equatorial (faces 4--7), and southern (faces 8--11)---each requiring different rotation transforms:

- **Northern faces:** Top and left neighbors are rotated by 90 deg and 180 deg respectively before extracting padding strips.
- **Equatorial faces:** Neighbors share the same orientation; no rotation is needed for most edges, but the top-left and bottom-right corners require special blending because three faces meet at these vertices. The blending averages the edge values of the two contributing faces at a 50/50 ratio.
- **Southern faces:** Bottom and right neighbors are rotated analogously to the northern case.

This topology-aware padding eliminates artificial discontinuities at face boundaries that would otherwise corrupt learned features.

**ConvNeXt blocks.** Each encoder and decoder level uses a ConvNeXt-style residual block [3]: two 3x3 HEALPix-padded convolutions with Capped GELU activations [2] (GELU clamped at a maximum value to prevent unbounded outputs during rollout), followed by a 1x1 projection, plus a residual skip connection (identity when channels match, otherwise a 1x1 convolution).

**Encoder--Decoder structure.** The encoder has three levels with channel widths (64, 128, 256), using 2x2 average pooling for downsampling. The decoder mirrors the encoder with transposed convolutions for upsampling and skip connections via channel-wise concatenation. A final 1x1 convolution projects back to the 84 output channels. The complete model has approximately **4.3 million** trainable parameters.

#### 2.2.2 3D U-Net (Baseline)

As a baseline, we trained a standard 3D U-Net [5] that treats the 12 HEALPix faces as a third spatial dimension. The encoder uses 3D convolutions with channel widths (32, 64, 128, 256) and max pooling with kernel (1, 2, 2) to preserve the face dimension. The decoder uses trilinear upsampling with skip connections. This architecture has approximately 30 million parameters and does not employ geometry-aware padding.

### 2.3 Training Protocol

Both models are trained to minimize the mean-squared error (MSE) between the predicted and actual reanalysis state at the next hourly time step (t + 1h). The loss is computed across all 84 channels (3 x 28 levels) and all 12 HEALPix faces.

For the HEALPix U-Net, we used the Adam optimizer [19] with learning rate 10^-4, batch size 1 (necessitated by the memory footprint at N_side = 64), and trained for 50 epochs on an 80/20 train/validation split. For the 3D U-Net baseline, we used learning rate 10^-3, batch size 4, and trained for 20 epochs.

After training, the model is used recursively for multi-step forecasting: the predicted field at time t + dt becomes the input to predict the field at t + 2dt, and so on up to 24 hours.

## 3. Results

### 3.1 Training Convergence

The HEALPix U-Net converges smoothly over 50 epochs. Starting from a validation MSE of 9.4 x 10^-4 after epoch 1, the loss decreases steadily to a best validation MSE of **1.58 x 10^-5** at epoch 48 (corresponding RMSE ~ 0.004 in normalized units). The training and validation curves track each other closely throughout, indicating no overfitting---a notable result given the model's 4.3M parameters and the relatively limited training data. The loss decreased by approximately two orders of magnitude during training, with the most rapid improvement occurring in the first 10 epochs.

| **Model** | **Params** | **HPX Level** | **Best Val. MSE** |
| :--- | :--- | :--- | :--- |
| 3D U-Net | ~30M | 3 (N_s=8) | 1.4 x 10^-4 |
| HPX U-Net | ~4.3M | 4 (N_s=16) | 1.4 x 10^-4 |
| HPX U-Net | ~4.3M | 6 (N_s=64) | **1.58 x 10^-5** |

*Table 1: Comparison of model architectures and their performance.*

The HEALPix U-Net at level 6 achieves an order-of-magnitude improvement in validation MSE over both the 3D U-Net baseline and the lower-resolution HEALPix U-Net, despite having nearly seven times fewer parameters than the 3D U-Net. We note that this comparison is not strictly controlled: the 3D U-Net was trained at lower resolution (level 3, N_side=8), with fewer epochs (20 vs. 50), and a different learning rate (10^-3 vs. 10^-4). The comparison is therefore intended to illustrate the qualitative advantage of geometry-aware convolutions at scale rather than a rigorous ablation. A controlled study isolating the individual contributions of HEALPix padding, resolution, and ConvNeXt blocks is planned for future work (Section 4.4).

### 3.2 Short-Range Prediction Skill

The trained HEALPix U-Net accurately reproduces the three-dimensional structure of Martian temperature and wind fields in single-step (one-hour-ahead) predictions. The model captures the large-scale thermal structure, including the strong temperature gradient across the polar regions, the warm equatorial band, and the cold southern polar vortex. The spatial correlation between prediction and ground truth is visually excellent, with the predicted field maintaining both the correct temperature range and the position of major features.

The one-hour RMSE of approximately 0.004 in normalized units is well below the typical hourly variance in the EMARS fields, indicating that the network has learned a meaningful dynamical propagation operator rather than a trivial identity mapping. A formal comparison against a persistence baseline (which assumes the state at t+1 is identical to t) and a climatological baseline is deferred to future work, along with evaluation using probabilistic scoring rules such as CRPS [24].

### 3.3 Recursive Rollout Stability

A critical test for any weather emulator is whether it remains stable when predictions are fed back as inputs for multi-step forecasting. We evaluated the HEALPix U-Net by unrolling it autoregressively for 24 consecutive one-hour steps starting from a single initial condition.

The error grows monotonically and smoothly from approximately 0.004 at T+1h to approximately 0.029 at T+24h. Crucially, the RMSE curve shows **no sign of divergence or exponential blowup** over the full 24-hour window. The growth rate decelerates after approximately 12 hours, suggesting that the model asymptotically approaches its internal climatology rather than producing unphysical states.

This 24-hour stable rollout represents a significant improvement over the baseline 3D U-Net, which exhibited forecast drift toward climatology by approximately 10 hours in preliminary experiments. The improvement is attributable to the HEALPix-aware padding, which prevents artificial discontinuities at face boundaries from amplifying during recursive application.

Despite 24 recursive applications, the predicted field retains the correct large-scale thermal structure: the position and intensity of the southern polar vortex, the warm equatorial band, and mid-latitude gradients are all preserved. Diamond-shaped artifacts corresponding to HEALPix face boundaries are visible in the 24-hour prediction, indicating that the inter-face padding, while effective, introduces subtle imprints under repeated application. These artifacts represent an area for architectural improvement.

### 3.4 Computational Efficiency

The HEALPix U-Net generates a one-hour forecast in approximately 0.5 seconds on a single NVIDIA GPU. Producing a full 24-hour rollout requires roughly 12 seconds of wall-clock time. In contrast, integrating the GFDL/NASA Mars GCM at comparable resolution requires substantial HPC resources and orders-of-magnitude more time [7]. This speedup makes the deep-learning emulator suitable for generating large ensembles for mission risk assessment, where hundreds of potential scenarios can be simulated in minutes.

The model's compact size (4.3M parameters, ~17 MB checkpoint) also makes it deployable on modest hardware, potentially even onboard future Mars-orbiting or surface platforms for real-time forecasting applications.

## 4. Discussion

### 4.1 Architecture Benefits

Our results confirm that respecting the geometry of the sphere during convolution yields substantial benefits for global weather emulation. The HEALPix-aware padding eliminates the artificial discontinuities that standard padding introduces at face boundaries. When the model is applied recursively, these discontinuities would otherwise amplify with each step, leading to the boundary artifacts and early divergence observed in naive approaches. The DLESyM-style architecture achieves better accuracy with 4.3M parameters than the 3D U-Net achieves with 30M, validating the principle that inductive biases aligned with the problem geometry can substitute for raw model capacity.

The Capped GELU activation [2] further contributes to rollout stability by preventing unbounded intermediate activations that can trigger numerical instability during long autoregressive chains.

### 4.2 Limitations

Several factors currently limit the model's performance and applicability.

**Normalized vs. physical-unit errors.** All errors in this study are reported in min-max normalized [0, 1] units. To provide physical context: Mars surface temperature in EMARS spans approximately 140--300 K (a range of ~160 K). An RMSE of 0.004 in normalized units thus corresponds to ~0.6 K at T+1h, while 0.029 corresponds to ~4.6 K at T+24h. For wind components with a typical range of ~200 m/s across all levels, these translate to ~0.8 m/s and ~5.8 m/s respectively. Future work should report per-variable, per-level errors in physical units for direct comparison with GCM skill scores and Earth ML weather model benchmarks.

**Forecast horizon.** While the model remains stable for 24 hours, the RMSE at T+24h (~0.029, or ~4.6 K for temperature) is approximately seven times larger than at T+1h (~0.004). Beyond ~12 hours, the forecast increasingly reflects the model's learned climatology rather than the specific atmospheric evolution from the initial condition. Extending the useful forecast horizon likely requires incorporating temporal context (e.g., multiple input time steps or recurrent architectures) and additional prognostic variables.

**Weather vs. climate timescales.** Our 24-hour (one Martian sol) stable rollout demonstrates weather-timescale forecasting skill, but climate emulation demands stable integration over thousands of sols spanning seasonal and inter-annual variability. The monotonic error growth observed suggests that the model would eventually saturate at its internal climatology rather than diverge, which is encouraging; however, maintaining physical fidelity over climate timescales will likely require enforcing conservation laws (mass, energy, angular momentum), coupling with surface and subsurface thermal models, and training on multi-year sequences spanning the full range of Martian seasons and dust storm activity.

**Aerosol representation.** Dust is a primary driver of the Martian atmospheric thermal structure [15, 16]. During global dust storms, radiative heating of the atmosphere changes drastically. Our model, which forecasts only T, U, and V, cannot anticipate these aerosol-driven regime transitions. Incorporating dust and water-ice opacity as additional channels is a critical next step.

**Face-boundary artifacts.** The diamond-shaped imprints visible in the 24-hour forecast indicate that the HEALPixPadding, while substantially better than naive alternatives, does not perfectly reproduce spherical continuity. The blending strategy at three-face vertices uses simple 50/50 averaging, which may introduce subtle biases under repeated application. More sophisticated blending or learned padding strategies could mitigate this.

**Reanalysis bias inheritance.** The model inherits biases from EMARS itself. Because EMARS assimilates observations from a limited number of local times (primarily corresponding to spacecraft overpasses), the diurnal cycle in the reanalysis may not be fully representative [8]. Future work could benefit from multi-model ensembles or physically-informed constraints enforcing conservation laws.

### 4.3 Broader Implications

By demonstrating that a parsimonious model can capture the core dynamics of a different planet's atmosphere, we validate the cross-planetary applicability of HEALPix-based architectures. Mars serves as a useful testbed: its lack of oceans and relatively simple surface isolate atmospheric behaviors that are often masked by complex oceanic interactions on Earth. Insights from the Martian digital twin---particularly regarding the minimum model complexity needed to represent atmospheric dynamics---could inform more efficient architectures for Earth climate emulators.

We envision a future system where a deep-learning digital twin is periodically updated with new spacecraft observations, allowing it to correct its forecasts toward the observed state, analogous to traditional Kalman filter-based systems but at a fraction of the computational cost.

### 4.4 Future Directions

Several avenues for extending this work follow naturally from the current limitations and recent advances in AI weather prediction:

1. **Probabilistic forecasting.** Replace the deterministic MSE objective with ensemble or distributional prediction heads, evaluated using proper scoring rules such as CRPS [24] and reliability diagrams, to quantify forecast uncertainty.

2. **Multi-step input conditioning.** Provide two or more consecutive time steps (t-1, t) as input, following GraphCast [13], to supply implicit velocity information and reduce the temporal context bottleneck.

3. **Alternative architectures.** Evaluate graph neural networks on an icosahedral multi-mesh [25], 3D attention mechanisms inspired by Pangu-Weather [12], and spherical Fourier neural operators [17] to benchmark against the HEALPix convolutional approach.

4. **Physics--ML hybrid coupling.** Adopt a NeuralGCM-style framework [23] in which a differentiable dynamical core handles resolved dynamics while learned parameterizations represent dust radiative transfer and subgrid processes.

5. **Dust and surface pressure channels.** Incorporate column-integrated dust optical depth [15] and surface pressure as additional prognostic variables to capture aerosol-driven regime transitions.

6. **Transfer learning from Earth.** Investigate whether pretraining on ERA5 reanalysis and finetuning on EMARS improves data efficiency, leveraging the shared fluid-dynamical principles across planetary atmospheres.

7. **Direct GCM comparison.** Generate parallel forecasts from the GFDL/NASA Mars GCM [7] at matched resolution and lead times for a rigorous skill comparison.

8. **Additional observational data.** Assimilate rover meteorological measurements from REMS (Curiosity) and MEDA (Perseverance) and additional satellite products to constrain the near-surface boundary layer.

9. **Controlled ablation study.** Isolate the individual contributions of HEALPix padding, ConvNeXt blocks, resolution, and CappedGELU activations through systematic ablations at fixed computational budget.

10. **Extended rollout.** Test multi-sol and seasonal-scale stability to assess suitability for climate applications, incorporating conservation constraints to prevent long-term drift.

## 5. Conclusions

We have presented a proof-of-concept deep-learning emulator for the Martian atmosphere that adapts the DLESyM HEALPix-aware convolutional framework to EMARS reanalysis data. Our key findings are:

1. **Geometry matters.** A HEALPix-aware 2D U-Net with inter-face padding achieves validation MSE of 1.58 x 10^-5---more than an order of magnitude better than a standard 3D U-Net baseline---with fewer parameters (4.3M vs. 30M).

2. **Stable long-range rollout.** The model produces physically coherent autoregressive forecasts for 24 hours with monotonically growing but bounded RMSE, extending the forecast horizon beyond the 10-hour limit of earlier approaches.

3. **Extreme efficiency.** One-hour forecasts require ~0.5 seconds on a single GPU, enabling ensemble forecasting for mission risk assessment at negligible computational cost compared to GCM integration.

4. **Compact deployability.** The 17 MB model checkpoint opens possibilities for onboard forecasting on future Mars missions.

While limitations regarding aerosol representation, face-boundary artifacts at long lead times, and reanalysis bias inheritance remain, the foundation laid here demonstrates that data-driven methods can capture the complex dynamical propagation of a non-terrestrial atmosphere. Future work will focus on incorporating aerosol dynamics, extending to multi-step input conditioning with recurrent units, exploring hybrid architectures coupling neural emulators with physical parameterizations, and validating against independent Mars atmospheric observations.

## Acknowledgments

We thank the Keck Institute for Space Studies (KISS) at the California Institute of Technology for organizing two workshops on "Digital Twins for Solar System Exploration: Enceladus," which provided the insight, expertise, and discussions that inspired this research. We thank the EMARS development team for creating and maintaining the publicly available reanalysis dataset [8]. This study was inspired by the parsimonious deep-learning weather prediction model developed by Karlbauer et al. [10] and the DLESyM framework [1]. We acknowledge the use of open-source packages including `earth2grid` [17], PyTorch [18], and HEALPix [11]. This work did not receive external funding.

## References

[1] Watt-Meyer, O., Dresdner, G., McGibbon, J., et al. (2024). ACE2: Accurately learning the Earth's climatic variables from a coupled atmosphere model. *arXiv preprint arXiv:2310.02074*.

[2] Hendrycks, D. and Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv preprint arXiv:1606.08415*.

[3] Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. (2022). A ConvNet for the 2020s. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11976--11986.

[4] Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *MICCAI*, pp. 234--241. Springer.

[5] Cicek, O., Abdulkadir, A., Lienkamp, S. S., Brox, T., and Ronneberger, O. (2016). 3D U-Net: Learning dense volumetric segmentation from sparse annotation. In *MICCAI*, pp. 424--432. Springer.

[6] Read, P. L., Lewis, S. R., and Mulholland, D. P. (2015). The physics of Martian weather and climate: A review. *Reports on Progress in Physics*, 78(12):125901.

[7] Haberle, R. M., Clancy, R. T., Forget, F., Smith, M. D., and Zurek, R. W., eds. (2017). *The Atmosphere and Climate of Mars*. Cambridge University Press.

[8] Greybush, S. J., Gillespie, H. E., and Wilson, R. J. (2019). Transient eddies in the TES/MCS Ensemble Mars Atmosphere Reanalysis System (EMARS). *Icarus*, 317:158--181.

[9] Greybush, S. J., Wilson, R. J., Hoffman, R. N., et al. (2012). Ensemble Kalman filter data assimilation of Thermal Emission Spectrometer temperature retrievals into a Mars GCM. *Journal of Geophysical Research: Planets*, 117(E11).

[10] Karlbauer, M., Cresswell-Clay, N., Durran, D. R., et al. (2024). Advancing parsimonious deep learning weather prediction using the HEALPix mesh. *Journal of Advances in Modeling Earth Systems*, 16(8):e2023MS004021.

[11] Gorski, K. M., Hivon, E., Banday, A. J., et al. (2005). HEALPix: A framework for high-resolution discretization and fast analysis of data distributed on the sphere. *The Astrophysical Journal*, 622(2):759.

[12] Bi, K., Xie, L., Zhang, H., et al. (2023). Pangu-Weather: A 3D high-resolution model for fast and accurate global weather forecast. *Nature*, 619(7970):533--538.

[13] Lam, R., Sanchez-Gonzalez, A., Willson, M., et al. (2023). GraphCast: Learning skillful medium-range global weather forecasting. *Science*, 382(6677):1416--1421.

[14] Pathak, J., Subramanian, S., Harrington, P., et al. (2022). FourCastNet: A global data-driven high-resolution weather forecasting model. *arXiv preprint arXiv:2202.11214*.

[15] Montabone, L., Forget, F., Millour, E., et al. (2015). Eight-year climatology of dust optical depth on Mars. *Icarus*, 251:65--95.

[16] Kahre, M. A., Murphy, J. R., Newman, C. E., et al. (2017). The Mars dust cycle. In Haberle, R. M. et al., eds., *The Atmosphere and Climate of Mars*, pp. 295--337. Cambridge University Press.

[17] Bonev, B., Kurth, T., Grossman, C., et al. (2023). Spherical Fourier neural operators: Learning stable dynamics on the sphere. In *ICML*, pp. 2806--2823. PMLR.

[18] Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In *NeurIPS*, vol. 32.

[19] Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. In *ICLR*.

[20] Battalio, J. M. and Wang, H. (2021). The Mars Dust Activity Database (MDAD): A comprehensive statistical study of dust storm sequences. *Icarus*, 354:114059.

[21] Rodriguez-Fernandez, N. J., Forget, F., Montabone, L., et al. (2023). Monitoring Mars dust storms with machine learning applied to Mars Climate Sounder observations. *Icarus*, 401:115596.

[22] Newman, C. E., Bertrand, T., Battalio, J. M., et al. (2021). Multi-model meteorological and aeolian predictions for Mars 2020 and the Jezero crater region. *Space Science Reviews*, 217(1):20.

[23] Kochkov, D., Yuval, J., Langmore, I., et al. (2024). Neural general circulation models for weather and climate. *Nature*, 632(8027):1060--1066.

[24] Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477):359--378.

[25] Keisler, R. (2022). Forecasting global weather with graph neural networks. *arXiv preprint arXiv:2202.07575*.
