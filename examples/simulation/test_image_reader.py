# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Simulation LOFAR imaging with Bipp (NUFFT).
"""

import sys
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
import bipp.filter
import time as tt
import matplotlib.pyplot as plt
import json

# Imaging
N_pix = 350

lsq_images = []
std_images = []
with bipp.ImageReader("image.h5") as reader:
    tags = reader.tags()
    for t in tags:
        if "lsq" in t:
            lsq_images.append(reader.read(t).reshape(N_pix, N_pix))
        elif "std" in t:
            std_images.append(reader.read(t).reshape(N_pix, N_pix))


lsq_images = np.array(lsq_images)
lsq_images = lsq_images.reshape((-1, N_pix, N_pix))

std_images = np.array(std_images)
std_images = std_images.reshape((-1, N_pix, N_pix))


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

# Data generation
T_integration = 8
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=40)
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=30)
time = obs_start + (T_integration * u.s) * np.arange(3595)
N_antenna = dev(time[0]).data.shape[0]
obs_end = time[-1]


lmn_grid, xyz_grid = frame.make_grids(N_pix, FoV, field_center)




I_lsq_eq = s2image.Image(lsq_images, xyz_grid)
I_std_eq = s2image.Image(std_images, xyz_grid)

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
    #  f"Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n"
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
)


#  plt.savefig("nufft_synthesis_std.png")
plt.figure()
titles = ["Strong sources", "Mild sources", "Faint Sources"]
for i in range(lsq_images.shape[0]):
    plt.subplot(1, lsq_images.shape[0], i + 1)
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
#  plt.savefig("nufft_synthesis_lsq.png")
plt.show()
