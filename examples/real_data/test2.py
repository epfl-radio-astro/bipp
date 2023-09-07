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
#ms_file = "/work/ska/MWA/1133149192-187-188_Sun_10s_cal.ms" # MWA Observation
#ms_file = "/home/krishna/OSKAR/Example/simulation_MWA_Obsparams.ms" # MWA Simulation

ms_file="/work/ska/RadioWeakLensing/MSFiles/WL_0.2h_noiseFalse.ms"
#ms_file="/work/ska/RadioWeakLensing/MSFiles/WL_1.0h_noiseFalse.ms"

wsclean_path = "/scratch/izar/krishna/MWA/WSClean/"
#wsclean_image = "1133149192-187-188_Sun_10s_cal_4_5_channels_weighting_natural-image.fits" # ONLY CHANNEL 5 MWA Observation
#wsclean_image = "1133149192-187-188_Sun_10s_cal1024_Pixels_0_64_channels-image.fits"
wsclean_image = "MWA_simulation_0_1_weighting_natural-image.fits"
#wsclean_image = "1133149192-187-188_Sun_10s_cal_0_64_channels_weighting_natural-image.fits" # ALL CHANNELS
wsclean_path += wsclean_image

output_dir = "/scratch/izar/krishna/bipp/bipp/"
################################################################################################################################################################################
# Control Variables ########################################################################################
###########################################################################################################

#Image params
N_pix = 2000
#N_pix = 1024

#Number of levels in output image
N_level = 3

filter_negative_eigenvalues = True

#clustering: If true will cluster log(eigenvalues) based on KMeans
clustering = True

# WSClean Grid: Use Coordinate grid from WSClean image if True
WSClean_grid = False

#ms_fieldcenter: Use field center from MS file if True; only invoked if WSClean_grid is False
ms_fieldcenter = True

# Field of View in degrees - only used when WSClean_grid is false
#FoV = np.deg2rad(6)
FoV = np.deg2rad(0.025)

# Column Name: Column in MS file to be imaged (DATA is usually uncalibrated, CORRECTED_DATA is calibration and MODEL_DATA contains WSClean model output)
column_name = "DATA"

# IF USING WSCLEAN IMAGE GRID: sampling wrt WSClean grid
# 1 means the output will have same number of pixels as WSClean image
# N means the output will have WSClean Image/N pixels
sampling = 1

# error tolerance for FFT
eps = 1e-3

#precision of calculation
precision = 'single'



#user_fieldcenter: Invoked if WSClean_grid and ms_fieldcenter are False - gives allows custom field center for imaging of specific region
user_fieldcenter = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")

#Time
time_start = 0
time_end = -1
time_slice = 1

# channel
channel_id = np.array([0], dtype = np.int64)
#channel_id = np.arange(64, dtype = np.int)

# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bipp.Context("CPU")

filter_tuple = ('lsq', 'std') # might need to make this a list

std_img_flag = True # put to true if std is passed as a filter

outName=output_dir + f"WL_10m_BB_{int(np.rad2deg(FoV)*60):2d}mFoV_Lvl{N_level}"
plotList= np.array([1,2,])
# 1 is lsq
# 2 is levels
# 3 is WSClean
# 4 is WSClean v/s lsq comparison

# 1 2 and 3 are on the same figure - we can remove 2 and 3 from this figure also
# 4 is on a different figure with 1 and 2

#######################################################################################################################################################
# Observation set up ########################################################################################
#######################################################################################################################################################


ms=measurement_set.SKALowMeasurementSet(ms_file)
N_station = 512 # change this to get this from measurement set
N_antenna = 512

#ms = measurement_set.MwaMeasurementSet(ms_file)
#N_antenna = 128 # change this to get this from measurement set
#N_station = 128

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
#opt.set_local_image_partition(bipp.Partition.none()) # Commented out
#opt.set_local_uvw_partition(bipp.Partition.none()) # Commented out
opt.set_local_image_partition(bipp.Partition.grid([8,8,1]))
opt.set_local_uvw_partition(bipp.Partition.grid([8,8,1]))
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
    num_time_steps = 0
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
        num_time_steps +=1

    print (f"Number of time steps: {num_time_steps}")
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
if (filter_negative_eigenvalues):
    I_lsq_eq = s2image.Image(lsq_image.reshape(int(N_level),lsq_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
else:
    I_lsq_eq = s2image.Image(lsq_image.reshape(int(N_level) + 1, lsq_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
print("lsq_image.shape =", lsq_image.shape)

if (std_img_flag):
    std_image = imager.get("STD").reshape((-1, N_pix, N_pix))
    if (filter_negative_eigenvalues):
        I_std_eq = s2image.Image(std_image.reshape(int(N_level), std_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
    else:
        I_std_eq = s2image.Image(std_image.reshape(int(N_level) + 1, std_image.shape[-2], std_image.shape[-1]), xyz_grid)
    print("std_image.shape =", std_image.shape)


# Without sensitivity imaging output

t2 = tt.time()

#plot output image

lsq_levels = I_lsq_eq.data # Nlevel, Npix, Npix
lsq_image = lsq_levels.sum(axis = 0)
print (f"Lsq Levels shape:{lsq_levels.shape}")

if (std_img_flag):
    std_levels = I_std_eq.data # Nlevel, Npix, Npix
    std_image = std_levels.sum(axis = 0)

    print (f"STD Levels shape:{std_levels.shape}")

WSClean_image = fits.getdata(wsclean_path)
WSClean_image = np.flipud(WSClean_image.reshape(WSClean_image.shape[-2:]))

if (filter_negative_eigenvalues):
    eigenlevels = N_level
else: 
    eigenlevels = N_level + 1

if (4 in plotList):
    if (filter_negative_eigenvalues):
        fig_comp, ax_comp = plt.subplots(int(len(filter_tuple)), 3, figsize = (40,20))
if (1 in plotList):
    fig_out, ax_out = plt.subplots(len(filter_tuple), 1, figsize = (40,20) )
if (2 in plotList):
    fig_out, ax_out = plt.subplots(len(filter_tuple), 1 + eigenlevels, figsize = (40,20))
if (3 in plotList):
    fig_out, ax_out = plt.subplots(len(filter_tuple), 2 + eigenlevels, figsize = (40,20))

if ((1 in plotList) or (2 in plotList) or (3 in plotList)):

    # Output LSQ Image

    BBScale = ax_out[0, 0].imshow(lsq_image, cmap = "cubehelix")
    ax_out[0, 0].set_title(r"LSQ IMG")
    divider = make_axes_locatable(ax_out[0, 0])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(BBScale, cax)

    # Output STD Image
    if (std_img_flag):
        BBScale = ax_out[1, 0].imshow(std_image, cmap = "cubehelix")
        ax_out[1, 0].set_title(r"STD IMG")
        divider = make_axes_locatable(ax_out[1, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(BBScale, cax)

    # output eigen levels
    if ((2 in plotList) or (3 in plotList)):
        for i in np.arange(eigenlevels):

            lsqScale = ax_out[0, i + 1].imshow(lsq_levels[i, :, :], cmap = "cubehelix", \
                        vmin = (lsq_levels[i, :, :].max() * std_levels[i, :, :].min()/std_levels[i, :, :].max() if std_img_flag else lsq_levels[i, :, :].min()))
            ax_out[0, i + 1].set_title(f"Lsq Lvl {i}")
            divider = make_axes_locatable(ax_out[0, i + 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(lsqScale, cax)

            if (std_img_flag):
                stdScale = ax_out[1, i + 1].imshow(std_levels[i, :, :], cmap = "cubehelix")
                ax_out[1, i + 1].set_title(f"Std Lvl {i}")
                divider = make_axes_locatable(ax_out[1, i + 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(stdScale, cax)

        # WSC Image
    if ((3 in plotList)):
        WSCleanScale = ax_out[0, -1].imshow(WSClean_image, cmap = "cubehelix")
        ax_out[0, -1].set_title(f"WSC IMG")
        divider = make_axes_locatable(ax_out[0, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(WSCleanScale, cax)
    
        if (std_img_flag):
            WSCleanScale = ax_out[1, -1].imshow(WSClean_image, cmap = "cubehelix")
            ax_out[1, -1].set_title(f"WSC IMG")
            divider = make_axes_locatable(ax_out[1, -1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(WSCleanScale, cax)

    fig_out.savefig(f"{outName}_{'Lvls' if (2 in plotList) else 'Summed'}")

# plot WSC, CASA and BB Comparisons here
if ((4 in plotList)):

    fig_comp, ax_comp = plt.subplots(len(filter_tuple), 3, figsize = (40, 30)) # Right now only WSC and BB included, have to include CASA

    # Comparison LSQ IMAGE 
    BBScale = ax_comp[0, 0].imshow(lsq_image, cmap = "RdBu_r")
    ax_comp[0, 0].set_title(r"LSQ IMG")
    divider = make_axes_locatable(ax_comp[0, 0])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(BBScale, cax)

    # Comparison WSClean image
    WSCleanScale = ax_comp[0, -2].imshow(WSClean_image, cmap='RdBu_r')
    ax_comp[0, -2].set_title(f"WSC IMG")
    divider = make_axes_locatable(ax_comp[0, -2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(WSCleanScale, cax)

    # Comparison STD image
    if (std_img_flag):
        BBScale = ax_comp[1, 0].imshow(std_image, cmap = "RdBu_r")
        ax_comp[1, 0].set_title(r"STD IMG")
        divider = make_axes_locatable(ax_comp[1, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(BBScale, cax)

        WSCleanScale = ax_comp[1, -2].imshow(WSClean_image, cmap='RdBu_r')
        ax_comp[1, -2].set_title(f"WSC IMG")
        divider = make_axes_locatable(ax_comp[1, -2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(WSCleanScale, cax)

    # Comparison LSQ-WSC Difference image
    diff_image = lsq_image - WSClean_image
    diff_norm = TwoSlopeNorm(vmin=diff_image.min(), vcenter=0, vmax=diff_image.max())

    diffScale = ax_comp[0, -1].imshow(diff_image, cmap = 'RdBu_r', norm=diff_norm)
    ax_comp[0, -1].set_title("Diff IMG")
    divider = make_axes_locatable(ax_comp[0, -1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(diffScale, cax)

    # Comparison LSQ/WSC - 1 image
    ratio_image = lsq_image/WSClean_image - 1
    clipValue = 2.5
    ratio_image = np.clip(ratio_image, -clipValue, clipValue)
    ratio_norm = TwoSlopeNorm(vmin=ratio_image.min(), vcenter=1, vmax=ratio_image.max())

    ratioScale = ax_comp[1, -1].imshow(ratio_image, cmap = 'RdBu_r', norm=ratio_norm)
    ax_comp[1, -1].set_title(f"Ratio IMG (clipped $\pm$ {clipValue})")
    divider = make_axes_locatable(ax_comp[1, -1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(ratioScale, cax)

    fig_comp.savefig(f"{outName}_comparison")

print(f'Elapsed time: {tt.time() - start_time} seconds.')