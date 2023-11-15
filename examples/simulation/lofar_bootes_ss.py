# #############################################################################
# lofar_bootes_ss.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Simulated LOFAR imaging with Bipp (StandardSynthesis).
"""

from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
import bipp.imot_tools.io.s2image as s2image
import bipp.imot_tools.math.sphere.grid as grid
import bipp.imot_tools.math.sphere.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import sys

import bipp.beamforming as beamforming
import bipp.gram as bb_gr
import bipp.parameter_estimator as bb_pe
import bipp.source as source
import bipp.statistics as statistics
import bipp.instrument as instrument
import bipp



def my_test(n, d):
    return I_est(n,d)
    #  d *= (d < 10.0) & (d > 2.0)
    #  return d


# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bipp.Context("AUTO")

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
FoV, frequency = np.deg2rad(5), 145e6
wl = constants.speed_of_light / frequency

# Instrument
N_station = 24
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock(ctx)

# Data generation
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=20)
vis = statistics.VisibilityGeneratorBlock(
    sky_model, T_integration, fs=196000, SNR=np.inf
)
time = obs_start + (T_integration * u.s) * np.arange(3595)
N_antenna = dev(time[0]).data.shape[0]

# Imaging
N_level = 2
precision = "single"
_, _, px_colat, px_lon = grid.equal_angle(
    N=dev.nyquist_rate(wl), direction=field_center.cartesian.xyz.value, FoV=FoV
)

px_grid = transform.pol2cart(1, px_colat, px_lon)
px_w = px_grid.shape[1]
px_h = px_grid.shape[2]
px_grid = px_grid.reshape(3, -1)
px_grid = px_grid / np.linalg.norm(px_grid, axis=0)

print("Image dimension = ", px_w, ", ", px_h)
print("precision = ", precision)
print("N_station = ", N_station)
print("N_antenna = ", N_antenna)
print("Proc = ", ctx.processing_unit)

### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95, ctx=ctx)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    I_est.collect(wl, S.data, W.data, XYZ.data)

# Imaging
imager = bipp.StandardSynthesis(
    ctx,
    N_level,
    ["LSQ", "STD"],
    px_grid[0],
    px_grid[1],
    px_grid[2],
    precision,
)

for t in ProgressBar(time[::25]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    imager.collect(wl, my_test, W.data, XYZ.data, S.data)

I_lsq = imager.get("LSQ").reshape((-1, px_w, px_h))
I_std = imager.get("STD").reshape((-1, px_w, px_h))

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
I_std_eq = s2image.Image(I_std, px_grid.reshape(3, px_w, px_h))
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
ax[0].set_title("Bipp Standardized Image")

I_lsq_eq = s2image.Image(I_lsq, px_grid.reshape(3, px_w, px_h))
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[1])
ax[1].set_title("Bipp Least-Squares Image")

fig.savefig("standard_synthesis.png")
plt.show()
