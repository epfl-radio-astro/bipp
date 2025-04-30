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
import bipp.selection as sel
import time as tt
import matplotlib.pyplot as plt


comm = bipp.communicator.world()

# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
#Note : When using MPI, mixing "CPU" and "GPU" on different ranks is possible.
ctx = bipp.Context("AUTO", comm)

# print build config
print("===== Config ====")
print("MPI = ", bipp.config.mpi)
print("OMP = ", bipp.config.omp)
print("CUDA = ", bipp.config.cuda)
print("ROCM = ", bipp.config.rocm)

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


# Imaging
N_pix = 350
N_level = 3
time_slice = 25

# image synthesis options
precision = "single"
tol = 1e-3

# file names
dataset_file = "test.h5"
image_prop_file = "image_prop.h5"
image_data_file = "image_data.h5"

print("N_pix = ", N_pix)
print("precision = ", precision)
print("N_station = ", N_station)
print("N_antenna = ", N_antenna)
print("Proc = ", ctx.processing_unit)

lmn_grid, xyz_grid = frame.make_grids(N_pix, FoV, field_center)

####################################################
# Create dataset by computing the eigendecomposition
####################################################
print("\n===== Create Dataset ====")
t1 = tt.time()
with bipp.DatasetFile.create(dataset_file, "lofar", N_antenna, N_station) as dataset:
    for t in ProgressBar(time[::time_slice]):
        XYZ = dev(t)
        UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
        W = mb(XYZ, wl)
        S = vis(XYZ, W, wl)
        uvw = frame.reshape_uvw(UVW_baselines_t)
        v, d, scale = bipp.eigh_gram(wl, S.data, W.data, XYZ.data)
        dataset.write(t.value, wl, scale, v, d, uvw)

t2 = tt.time()
print(f"Elapsed time: {t2 - t1} seconds.")

#################################################
# Estimate parameters and generate a selection
#################################################
print("\n===== Parameter Estimation ====")
t1 = tt.time()
selection = {}
I_est = bb_pe.ParameterEstimator(N_level, sigma=0.95)
with bipp.DatasetFile.open(dataset_file) as dataset:
    # collect eigenvalues
    for idx in range(0, dataset.num_samples()):
        I_est.collect(dataset.eig_val(idx))

    # compute intervals to partition eigenvalues
    intervals = I_est.infer_parameters()

    # generate selection for lsq and std filters with
    # computed intervals
    filters = ["lsq", "std"]

    for filter_name in filters:
        for level in range(intervals.shape[0]):
            fi = bipp.filter.Filter(filter_name, intervals[level, 0], intervals[level,1])
            tag = f"{filter_name}_level_{level}"
            level_selection = {}
            for idx in range(0, dataset.num_samples()):
                level_selection[idx] = fi(dataset.eig_val(idx))
            selection[tag] = level_selection

# optionally export selection for use with 'bipp_synthesis' executable
sel.export_selection(selection, 'selection.json')

t2 = tt.time()
print(f"Elapsed time: {t2 - t1} seconds.")

#################################################
# Create image property file with lmn coordinates
#################################################
print("\n===== Image property generation ====")
t1 = tt.time()
with bipp.ImagePropFile.create(image_prop_file, lmn_grid.transpose()) as image_prop:
    # we can optionally write meta data for later use like plotting
    image_prop.set_meta("fov", FoV)

t2 = tt.time()
print(f"Elapsed time: {t2 - t1} seconds.")

#################################################
# Compute image synthesis for given selection
#################################################
print("\n===== Image Synthesis ====")
t1 = tt.time()

# Nufft Synthesis options
opt = bipp.NufftSynthesisOptions()
opt.set_tolerance(tol)
opt.set_precision(precision)
# Set the domain splitting methods for uvw coordinates.
# Splitting decreases memory usage, but may lead to lower performance.
# Best used with a wide spread of image or uvw coordinates.
# Possible options are "grid", "none" or "auto"
opt.set_local_uvw_partition(bipp.Partition.auto())

with bipp.DatasetFile.open(dataset_file) as dataset, bipp.ImagePropFile.open(image_prop_file) as image_prop:
    bipp.image_synthesis(ctx, opt, dataset, selection, image_prop, image_data_file)

t2 = tt.time()
print(f"Elapsed time: {t2 - t1} seconds.")

####################################################
# Process images
####################################################
print("\n===== Process Images ====")
t1 = tt.time()
lsq_images = []
std_images = []
with bipp.ImageDataFile.open(image_data_file) as reader:
    tags = reader.tags()
    tags.sort()
    for t in tags:
        if "lsq" in t:
            lsq_images.append(reader.get(t).reshape(N_pix, N_pix))
        elif "std" in t:
            std_images.append(reader.get(t).reshape(N_pix, N_pix))


lsq_images = np.array(lsq_images)
lsq_images = lsq_images.reshape((-1, N_pix, N_pix))

std_images = np.array(std_images)
std_images = std_images.reshape((-1, N_pix, N_pix))


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
    f"Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n"
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
)

plt.savefig("nufft_synthesis_std.png")
plt.figure()
titles = ["Strong sources", "Mild sources", "Faint Sources"]
for i in range(lsq_images.shape[0]):
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

t2 = tt.time()
print(f"Elapsed time: {t2 - t1} seconds.")
