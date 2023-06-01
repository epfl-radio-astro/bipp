# #############################################################################
# lofar_bootes_nufft.py
# ======================
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]

## TIME SLICE QUERY 
## num antennas, num stations????
# #############################################################################

"""
Simulation LOFAR imaging with Bipp (NUFFT).
"""

from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.io.fits as fits
import astropy.wcs as awcs
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
import bipp.measurement_set as measurement_set
import time as tt
import matplotlib.pyplot as plt

start_time= tt.time()
################################################################################################################################################################################
## INPUT VARIABLES ################################################################################################################################################################################
################################################################################################################################################################################

ms_file = "/scratch/izar/krishna/MWA/MS_Files/1133149192-187-188_Sun_10s_cal.ms"
wsclean_path = "/scratch/izar/krishna/MWA/WSClean/"
wsclean_image = "1133149192-187-188_Sun_10s_cal_4_5_channels_weighting_natural-image.fits" # ONLY CHANNEL 5
#wsclean_image = "1133149192-187-188_Sun_10s_cal_0_64_channels_weighting_natural-image.fits" # ALL CHANNELS
wsclean_path += wsclean_image

output_dir = "/scratch/izar/krishna/bipp/"
################################################################################################################################################################################
# Control Variables ########################################################################################
###########################################################################################################

#Image params
N_pix = 1024

# error tolerance for FFT
eps = 1e-3

#precision of calculation
precision = 'double'

# Field of View in degrees - only used when WSClean_grid is false
FoV = np.deg2rad(6)

#Number of levels in output image
N_level = 4

#clustering: If true will cluster log(eigenvalues) based on KMeans
clustering = True

# IF USING WSCLEAN IMAGE GRID: sampling wrt WSClean grid
# 1 means the output will have same number of pixels as WSClean image
# N means the output will have WSClean Image/N pixels
sampling = 1

# Column Name: Column in MS file to be imaged (DATA is usually uncalibrated, CORRECTED_DATA is calibration and MODEL_DATA contains WSClean model output)
column_name = "DATA"

# WSClean Grid: Use Coordinate grid from WSClean image if True
WSClean_grid = False

#ms_fieldcenter: Use field center from MS file if True; only invoked if WSClean_grid is False
ms_fieldcenter = True

#user_fieldcenter: Invoked if WSClean_grid and ms_fieldcenter are False - gives allows custom field center for imaging of specific region
user_fieldcenter = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")

#Time
time_start = 0
time_end = -1
time_slice = 1

# channel
channel_id = np.array([4], dtype = np.int64)
#channel_id = np.arange(64, dtype = np.int)

# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bipp.Context("AUTO")

#######################################################################################################################################################
# Observation set up ########################################################################################
#######################################################################################################################################################

ms = measurement_set.MwaMeasurementSet(ms_file)
N_station = 128 # change this to get this from measurement set
N_antenna = 128 # change this to get this from measurement set

try:
    if (channel_id.shape[0] > 1):
        frequency = ms.channels["FREQUENCY"][0] + (ms.channels["FREQUENCY"][-1] - ms.channels["FREQUENCY"][0])/2
        print ("Multi-channel mode with ", channel_id.shape[0], "channels.")
    else: 
        frequency = ms.channels["FREQUENCY"][channel_id]
        print ("Single channel mode.")
except:
    frequency = ms.channels["FREQUENCY"][channel_id]
    print ("Single channel mode.")

wl = constants.speed_of_light / frequency.to_value(u.Hz) [0]
print (f"wl:{wl}; f: {frequency}")

if (WSClean_grid): 
    with fits.open(wsclean_path, mode="readonly", memmap=True, lazy_load_hdus=True) as hdulist:
        cl_WCS = awcs.WCS(hdulist[ext].header)
        cl_WCS = cl_WCS.sub(['celestial'])
        cl_WCS = cl_WCS.slice((slice(None, None, sampling), slice(None, None, sampling)))

    field_center = ms.field_center

    width_px, height_px= 2*cl_WCS.wcs.crpix 
    cdelt_x, cdelt_y = cl_WCS.wcs.cdelt 
    FoV = np.deg2rad(abs(cdelt_x*width_px) )
    print ("WSClean Grid used.")
else: 
    if (ms_fieldcenter):
        field_center = ms.field_center
        print ("Self generated grid used based on ms fieldcenter")
    else:
        field_center = user_fieldcenter
        print ("Self generated grid used based on user fieldcenter")

lmn_grid, xyz_grid = frame.make_grids(N_pix, FoV, field_center)

gram = bb_gr.GramBlock(ctx)

print (f"Initial set up takes {tt.time() - start_time} s")

# Nufft Synthesis options

opt = bipp.NufftSynthesisOptions()
# Set the tolerance for NUFFT, which is the maximum relative error.
opt.set_tolerance(eps)
# Set the maximum number of data packages that are processed together after collection.
# A larger number increases memory usage, but usually improves performance.
# If set to "None", an internal heuristic will be used.
opt.set_collect_group_size(None)
# Set the domain splitting methods for image and uvw coordinates.
# Splitting decreases memory usage, but may lead to lower performance.
# Best used with a wide spread of image or uvw coordinates.
# Possible options are "grid", "none" or "auto"
opt.set_local_image_partition(bipp.Partition.grid([1,1,1]))
opt.set_local_uvw_partition(bipp.Partition.none())

t1 = tt.time()
#time_slice = 25 ### why is this 25 - ask simon @@@

print("N_pix = ", N_pix)
print("precision = ", precision)
print("Proc = ", ctx.processing_unit)

print (f"Initial set up takes {tt.time() - start_time} s")


########################################################################################
### Intensity Field ########################################################################################
########################################################################################
# Parameter Estimation
########################################################################################

if (clustering):
    I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=1, ctx=ctx)
    for t, f, S, uvw_t in ProgressBar(
            ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW = True)
    ):
        wl = constants.speed_of_light / f.to_value(u.Hz)
        XYZ = ms.instrument(t)

        W = ms.beamformer(XYZ, wl)
        G = gram(XYZ, W, wl)
        S, _ = measurement_set.filter_data(S, W)
        I_est.collect(S, G)

    N_eig, intensity_intervals = I_est.infer_parameters()
else:
    # Set number of eigenvalues to number of eigenimages 
    # and equally divide the data between them 
    N_eig, intensity_intervals = N_level, np.arange(N_level)

print(f"Clustering: {clustering}")
print (f"Number of Eigenvalues:{N_eig}, Intensity intervals: {intensity_intervals}")

########################################################################################
# Imaging ########################################################################################
########################################################################################
imager = bipp.NufftSynthesis(
    ctx,
    opt,
    N_antenna,
    N_station,
    intensity_intervals.shape[0],
    ["LSQ", "SQRT"],
    lmn_grid[0],
    lmn_grid[1],
    lmn_grid[2],
    precision,
)
for t, f, S, uvw_t in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW = True)
):
    
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    
    S, W = measurement_set.filter_data(S, W)
    
    UVW_baselines_t = uvw_t
    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    
    imager.collect(N_eig, wl, intensity_intervals, W.data, XYZ.data, uvw, S.data)
    
lsq_image = imager.get("LSQ").reshape((-1, N_pix, N_pix))
sqrt_image = imager.get("SQRT").reshape((-1, N_pix, N_pix))
########################################################################################
### Sensitivity Field ########################################################################################
########################################################################################
# Parameter Estimation ########################################################################################
########################################################################################

if (clustering):
    S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=1, ctx=ctx)
    for t, f, S in ProgressBar(ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name)):
        wl = constants.speed_of_light / f.to_value(u.Hz)
        XYZ = ms.instrument(t)
        W = ms.beamformer(XYZ, wl)
        G = gram(XYZ, W, wl)

        S_est.collect(G)
    N_eig = S_est.infer_parameters()
else:
    N_eig=N_level

########################################################################################
# Imaging ########################################################################################
########################################################################################

sensitivity_intervals = np.array([[0, np.finfo("f").max]])
imager = None  # release previous imager first to some additional memory
imager = bipp.NufftSynthesis(
    ctx,
    opt,
    N_antenna,
    N_station,
    sensitivity_intervals.shape[0],
    ["INV_SQ"],
    lmn_grid[0],
    lmn_grid[1],
    lmn_grid[2],
    precision,
)
for t, f, S, uvw_t in ProgressBar(
        ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW = True)
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)

    _, W = measurement_set.filter_data(S, W)

    uvw = frame.reshape_and_scale_uvw(wl, uvw_t)
    imager.collect(N_eig, wl, sensitivity_intervals, W.data, XYZ.data, uvw, None)

sensitivity_image = imager.get("INV_SQ").reshape((-1, N_pix, N_pix))

# Plot Results ================================================================
I_lsq_eq = s2image.Image(lsq_image / sensitivity_image, xyz_grid)
I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, xyz_grid)
t2 = tt.time()
print(f"Elapsed time: {t2 - t1} seconds.")

plt.figure()
ax = plt.gca()
I_lsq_eq.draw(
    ax=ax,
    data_kwargs=dict(cmap="cubehelix"),
    show_gridlines=False,
)
ax.set_title(
    f" Bipp least-squares, sensitivity-corrected image (NUFFT)\n"
    f" FoV: {np.round(FoV * 180 / np.pi)} degrees.\n"
    f" Run time {np.floor(t2 - t1)} seconds."
)

plt.figure()
ax = plt.gca()
I_sqrt_eq.draw(
    ax=ax,
    data_kwargs=dict(cmap="cubehelix"),
    show_gridlines=False,
)
ax.set_title(
    f"Bipp sqrt, sensitivity-corrected image (NUFFT)\n"
    f"FoV: {np.round(FoV * 180 / np.pi)} degrees.\n"
    f"Run time {np.floor(t2 - t1)} seconds."
)

plt.figure()
titles = ["Strong sources", "Mild sources", "Faint Sources"]
titles = titles [:N_level]
for i in range(lsq_image.shape[0]):
    plt.subplot(1, N_level, i + 1)
    ax = plt.gca()
    plt.title(titles[i])
    I_lsq_eq.draw(
        index=i,
        ax=ax,
        data_kwargs=dict(cmap="cubehelix"),
        show_gridlines=False,
    )

plt.suptitle(f"Bipp Eigenmaps")
#  plt.show()
plt.savefig("final_bb.png")
