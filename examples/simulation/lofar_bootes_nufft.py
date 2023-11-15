# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Simulation LOFAR imaging with Bipp (NUFFT).
"""

from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import bipp.imot_tools.io.s2image as s2image
import numpy as np
import scipy.constants as constants
import bipp
from bipp.imot_tools.io.plot import cmap
import bipp.beamforming as beamforming
import bipp.gram as bb_gr
import bipp.parameter_estimator as bb_pe
import bipp.source as source
import bipp.instrument as instrument
import bipp.frame as frame
import bipp.statistics as statistics
import time as tt
import matplotlib.pyplot as plt


# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bipp.Context("CPU")

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
FoV, frequency = np.deg2rad(10), 145e6
wl = constants.speed_of_light / frequency

# Instrument
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock(ctx)

# Data generation
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=40)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time = obs_start + (T_integration * u.s) * np.arange(3595)
N_antenna = dev(time[0]).data.shape[0]
obs_end = time[-1]

# Nufft Synthesis options

opt = bipp.NufftSynthesisOptions()
# Set the tolerance for NUFFT, which is the maximum relative error.
opt.set_tolerance(1e-3)
# Set the maximum number amount of memory to be used for collecting time step data.
# A larger number increases memory usage, but usually improves performance.
opt.set_collect_memory(0.2)
# Set the domain splitting methods for image and uvw coordinates.
# Splitting decreases memory usage, but may lead to lower performance.
# Best used with a wide spread of image or uvw coordinates.
# Possible options are "grid", "none" or "auto"
opt.set_local_image_partition(bipp.Partition.grid([1,1,1]))
opt.set_local_uvw_partition(bipp.Partition.none())
precision = "single"


# Imaging
N_pix = 350

t1 = tt.time()
N_level = 3
time_slice = 25

print("N_pix = ", N_pix)
print("precision = ", precision)
print("N_station = ", N_station)
print("N_antenna = ", N_antenna)
print("Proc = ", ctx.processing_unit)

lmn_grid, xyz_grid = frame.make_grids(N_pix, FoV, field_center)

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95, ctx=ctx)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    I_est.collect(wl, S.data, W.data, XYZ.data)

# Imaging
imager = bipp.NufftSynthesis(
    ctx,
    opt,
    N_level,
    ["LSQ", "STD"],
    lmn_grid[0],
    lmn_grid[1],
    lmn_grid[2],
    precision,
)

for t in ProgressBar(time[::time_slice]):
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    imager.collect(wl, I_est, W.data, XYZ.data, uvw, S.data)

lsq_image = imager.get("LSQ").reshape((-1, N_pix, N_pix))
std_image = imager.get("STD").reshape((-1, N_pix, N_pix))

# Plot Results ================================================================
#  fig, ax = plt.subplots(ncols=2)
#  I_std_eq = s2image.Image(I_std, xyz_grid)
#  I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
#  I_std_eq.draw(
#      catalog=sky_model.xyz.T,
#      ax=ax,
#      data_kwargs=dict(cmap="cubehelix"),
#      show_gridlines=False,
#      catalog_kwargs=dict(s=30, linewidths=0.5, alpha=0.5),
#  )
#  ax[0].set_title("Bipp Standardized Image")

#  I_lsq_eq = s2image.Image(I_lsq, xyz_grid)
#  I_lsq_eq.draw(
#      catalog=sky_model.xyz.T,
#      ax=ax,
#      data_kwargs=dict(cmap="cubehelix"),
#      show_gridlines=False,
#      catalog_kwargs=dict(s=30, linewidths=0.5, alpha=0.5),
#  )
#  ax[1].set_title("Bipp Least-Squares Image")

#  fig.savefig("nufft_synthesis.png")
#  plt.show()

#  exit(0)

I_lsq_eq = s2image.Image(lsq_image, xyz_grid)
I_std_eq = s2image.Image(std_image, xyz_grid)
t2 = tt.time()
print(f"Elapsed time: {t2 - t1} seconds.")

plt.figure()
ax = plt.gca()
I_lsq_eq.draw(
    catalog=sky_model.xyz.T,
    ax=ax,
    data_kwargs=dict(cmap="cubehelix"),
    show_gridlines=False,
    catalog_kwargs=dict(s=30, linewidths=0.5, alpha=0.5),
)
ax.set_title(
    f"Bipp least-squares, sensitivity-corrected image (NUFFT)\n"
    f"Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n"
    f"Run time {np.floor(t2 - t1)} seconds."
)

plt.figure()
ax = plt.gca()
I_std_eq.draw(
    catalog=sky_model.xyz.T,
    ax=ax,
    data_kwargs=dict(cmap="cubehelix"),
    show_gridlines=False,
    catalog_kwargs=dict(s=30, linewidths=0.5, alpha=0.5),
)
ax.set_title(
    f"Bipp STD, sensitivity-corrected image (NUFFT)\n"
    f"Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n"
    f"Run time {np.floor(t2 - t1)} seconds."
)

plt.savefig("nufft_synthesis_std.png")
plt.figure()
titles = ["Strong sources", "Mild sources", "Faint Sources"]
for i in range(lsq_image.shape[0]):
    plt.subplot(1, N_level, i + 1)
    ax = plt.gca()
    plt.title(titles[i])
    I_lsq_eq.draw(
        index=i,
        catalog=sky_model.xyz.T,
        ax=ax,
        data_kwargs=dict(cmap="cubehelix"),
        catalog_kwargs=dict(s=30, linewidths=0.5, alpha=0.5),
        show_gridlines=False,
    )

plt.suptitle(f"Bipp Eigenmaps")
plt.savefig("nufft_synthesis_lsq.png")
plt.show()
