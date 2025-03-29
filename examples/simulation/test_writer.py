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
import bipp.parameter_estimator as bb_pe
import bipp.source as source
import bipp.instrument as instrument
import bipp.frame as frame
import bipp.statistics as statistics
import bipp.filter
import time as tt
import matplotlib.pyplot as plt


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


t1 = tt.time()
N_level = 3
time_slice = 25


XYZ = dev(time[0])
W = mb(XYZ, wl)

comm = bipp.communicator.world()
ctx = bipp.Context("AUTO", comm)

with bipp.DatasetFile.create("test.h5", "lofar", W.data.shape[0], W.data.shape[1]) as dataset:
    for t in ProgressBar(time[::time_slice]):
        XYZ = dev(t)
        UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
        W = mb(XYZ, wl)
        S = vis(XYZ, W, wl)
        uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
        v, d, scale =bipp.eigh(ctx, wl, S.data, W.data, XYZ.data)
        dataset.write(wl, scale, v, d, XYZ.data, uvw)
        #  dataset.process_and_write('single', wl, S.data, W.data, XYZ.data, uvw)
