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
import sys
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
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

start_time= tt.time()
################################################################################################################################################################################
## INPUT VARIABLES ################################################################################################################################################################################
################################################################################################################################################################################

#ms_file = "/scratch/izar/krishna/MWA/MS_Files/1133149192-187-188_Sun_10s_cal.ms"
ms_file = "/work/ska/MWA/1133149192-187-188_Sun_10s_cal.ms"

wsclean_path = "/scratch/izar/krishna/MWA/WSClean/"
wsclean_image = "1133149192-187-188_Sun_10s_cal_4_5_channels_weighting_natural-image.fits" # ONLY CHANNEL 5
#wsclean_image = "1133149192-187-188_Sun_10s_cal1024_Pixels_0_64_channels-image.fits"

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
precision = 'single'

# Field of View in degrees - only used when WSClean_grid is false
FoV = np.deg2rad(6)

#Number of levels in output image
N_level = 3

filter_negative_eigenvalues = False

#clustering: If true will cluster log(eigenvalues) based on KMeans
clustering = True

# IF USING WSCLEAN IMAGE GRID: sampling wrt WSClean grid
# 1 means the output will have same number of pixels as WSClean image
# N means the output will have WSClean Image/N pixels
sampling = 1

# Column Name: Column in MS file to be imaged (DATA is usually uncalibrated, CORRECTED_DATA is calibration and MODEL_DATA contains WSClean model output)
column_name = "DATA"

# WSClean Grid: Use Coordinate grid from WSClean image if True
WSClean_grid = True

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
ctx = bipp.Context("CPU")

filter_tuple = ('lsq', 'std') # might need to make this a list

std_img_flag = True # put to true if std is passed as a filter

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
        cl_WCS = awcs.WCS(hdulist[0].header)
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
#opt.set_local_image_partition(bipp.Partition.grid([1,1,1]))
opt.set_local_image_partition(bipp.Partition.none())
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
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PARAMETER ESTIMATION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
if (clustering):
    I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=1, ctx=ctx,
                                                   filter_negative_eigenvalues=filter_negative_eigenvalues)
    #for t, f, S, uvw_t in ProgressBar(
    for t, f, S in ProgressBar(
            #ms.visibilities(channel_id=[channel_id], time_id=slice(time_start, time_end, time_slice), column=column_name, return_UVW = True)
            ms.visibilities(channel_id=channel_id, time_id=slice(time_start, time_end, time_slice), column=column_name)
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
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ IMAGING @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
imager = bipp.NufftSynthesis(
    ctx,
    opt,
    N_antenna,
    N_station,
    intensity_intervals.shape[0],
    filter_tuple,
    lmn_grid[0],
    lmn_grid[1],
    lmn_grid[2],
    precision,
    filter_negative_eigenvalues
)

for t, f, S in ProgressBar(
        ms.visibilities(channel_id=channel_id, time_id=slice(time_start, time_end, time_slice), column=column_name)
):
    
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    
    S, W = measurement_set.filter_data(S, W)
    
    #UVW_baselines_t = uvw_t
    #uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    UVW_baselines_t = ms.instrument.baselines(t, uvw=True, field_center=ms.field_center)
    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)

    imager.collect(N_eig, wl, intensity_intervals, W.data, XYZ.data, uvw, S.data)

lsq_image = imager.get("LSQ").reshape((-1, N_pix, N_pix))
I_lsq_eq = s2image.Image(lsq_image.reshape(int(N_level) + 1, lsq_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
print("lsq_image.shape =", lsq_image.shape)

if (std_img_flag):
    std_image = imager.get("STD").reshape((-1, N_pix, N_pix))
    I_std_eq = s2image.Image(std_image.reshape(int(N_level) + 1, std_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
    print("lsq_image.shape =", lsq_image.shape)


# Without sensitivity imaging output

t2 = tt.time()

#plot output image

fig, ax = plt.subplots(int(len(filter_tuple)), N_level + 4, figsize = (40, 20))


lsq_levels = I_lsq_eq.data # Nlevel, Npix, Npix
std_levels = I_std_eq.data # Nlevel, Npix, Npix

lsq_image = lsq_levels.sum(axis = 0)
std_image = std_levels.sum(axis = 0)

BBScale = ax[0, 0].imshow(lsq_image, cmap = "RdBu_r")
ax[0, 0].set_title(r"LSQ IMG")
divider = make_axes_locatable(ax[0, 0])
cax = divider.append_axes("right", size = "5%", pad = 0.05)
cbar = plt.colorbar(BBScale, cax)

#plot WSClean image
WSClean_image = fits.getdata(wsclean_path)
WSClean_image = np.flipud(WSClean_image.reshape(WSClean_image.shape[-2:]))
WSCleanScale = ax[0, -2].imshow(WSClean_image, cmap='RdBu_r')
ax[0, -2].set_title(f"WSC IMG")
divider = make_axes_locatable(ax[0, -2])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(WSCleanScale, cax)

if (std_img_flag):
    BBScale = ax[1, 0].imshow(std_image, cmap = "RdBu_r")
    ax[1, 0].set_title(r"STD IMG")
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(BBScale, cax)

    WSCleanScale = ax[1, -2].imshow(WSClean_image, cmap='RdBu_r')
    ax[1, -2].set_title(f"WSC IMG")
    divider = make_axes_locatable(ax[1, -2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(WSCleanScale, cax)

for i in np.arange(N_level + 1):
    lsqScale = ax[0, i + 1].imshow(lsq_levels[i, :, :], cmap = 'RdBu_r')
    ax[0, i + 1].set_title(f"Lsq Lvl {i}")
    divider = make_axes_locatable(ax[0, i + 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(lsqScale, cax)

    if (std_img_flag):
        stdScale = ax[1, i + 1].imshow(std_levels[i, :, :], cmap = 'RdBu_r')
        ax[1, i + 1].set_title(f"Std Lvl {i}")
        divider = make_axes_locatable(ax[1, i + 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(stdScale, cax)

diff_image = lsq_image - WSClean_image
diff_norm = TwoSlopeNorm(vmin=diff_image.min(), vcenter=0, vmax=diff_image.max())

diffScale = ax[0, -1].imshow(diff_image, cmap = 'RdBu_r', norm=diff_norm)
ax[0, -1].set_title("Diff IMG")
divider = make_axes_locatable(ax[0, -1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(diffScale, cax)

ratio_image = lsq_image/WSClean_image
ratio_image = np.clip(ratio_image, -2.5, 2.5)
ratio_norm = TwoSlopeNorm(vmin=ratio_image.min(), vcenter=1, vmax=ratio_image.max())

ratioScale = ax[1, -1].imshow(ratio_image, cmap = 'RdBu_r', norm=ratio_norm)
ax[1, -1].set_title("Ratio IMG")
divider = make_axes_locatable(ax[1, -1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(ratioScale, cax)

plt.savefig("spk_mwa_simulation_normalisedish.png")

print(f'Elapsed time: {tt.time() - start_time} seconds.')