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
# for final figure text sizes
plt.rcParams.update({'font.size': 50})



start_time= tt.time()
#####################################################################################################################################################################
# Input and Control Variables #####################################################################################################################################################################
#####################################################################################################################################################################
try:
    telescope_name =sys.argv[1]
    ms_file = sys.argv[2]
    filter_negative_eigenvalues= True

    if (telescope_name.lower()=="skalow"):
        ms=measurement_set.SKALowMeasurementSet(ms_file)
        N_station = 512
        N_antenna = 512
        

    elif (telescope_name.lower()=="mwa"):
        ms = measurement_set.MwaMeasurementSet(ms_file)
        N_station = 128
        N_antenna = 128

    elif (telescope_name.lower()=="lofar"):
        N_station = 37 # netherlands, 52 international
        N_antenna = 37 # netherlands, 52 international
        ms = measurement_set.LofarMeasurementSet(ms_file, N_station = N_station, station_only=True)

    else: 
        raise(NotImplementedError("A measurement set class for the telescope you are searching for has not been implemented yet - please feel free to implement the class yourself!"))

except: 
    raise(SyntaxError("This file must be called with the telescope name and path to ms file at a minimum. "+\
                      "Eg:\npython realData.py [telescope name(string)] [path_to_ms(string)] [output_name(string)] [N_pix(int)] [FoV(float(deg))] [N_levels(int)] [Clustering(bool/list_of_eigenvalue_bin_edges)] [partitions] [WSCleangrid(bool)] [WSCleanPath(string)]"))

# EXAMPLE COMMAND LINE RUNS: 
# python ~/bipp/bipp/examples/real_data/realData.py LOFAR ~/bluebild/testData/gauss4_t201806301100_SBL180.MS 4Gauss_t1_sigma095 2000 4.26 4 3e24,7e3,0 2>&1 | tee gauss4_sigma095_t1.log
# python ~/bipp/bipp/examples/real_data/realData.py LOFAR ~/bluebild/testData/gauss4_t201806301100_SBL180.MS 4Gauss_t1_sigma095_custom 2000 4.26 4 3e34,7e4,4e3,2e3,0 2>&1 | tee gauss4_sigma095_t1_custom.log
#
try:
    outName=sys.argv[3]
except:
    outName="test"
try:
    N_pix=int(sys.argv[4])
except:
    N_pix=2000 #2000 for LOFAR #512/1024/2048 for MWA
try:
    FoV = np.deg2rad(float (sys.argv[5]))
except:
    FoV = np.deg2rad(6) #4.26 deg for Lofar, 13.1 for Mwa
"""
 # time and channel id stuff to be implemented!!!
try:
    try:
        channel_id=bool(sys.argv[6])
        if (channel_id==True):
            # put code to read all channels
            channel_id = np.array()
    else:
        channelIDstr=sys.argv[10].split(",")
        channels=[]
        for channelID in channelIDstr:
            channel_id.append(int(channelID))
        channel_id = np.array(channel_id)


try:
    N_level = int(sys.argv[8])
except:
    N_level=3
try:
    try:
        clustering = bool(sys.argv[9]) # True or set of numbers which act as bins,separated by commas and NO spaces
        clusteringBool = True
    except:
        binEdgesStr = sys.argv[10].split(",")
        clustering = []
        for binEdge in binEdgesStr:
            clustering.append(float(binEdge))
        clustering= np.array(clustering)
        clusteringBool = False
"""
try:
    N_level = int(sys.argv[6])
except:
    N_level=4 # 4 for lofar 4 gauss
try:
    try:
        clusterEdges = np.array(sys.argv[7].split(","), dtype=np.float32)
        clusteringBool = False
        binStart = clusterEdges[0]
        clustering = []
        for binEdge in clusterEdges[1:]:
            binEnd = binEdge
            clustering.append([binEnd, binStart])
            binStart = binEnd
        clustering = np.asarray(clustering, dtype=np.float32)
    except:
        clustering = bool(sys.argv[7]) # True or set of numbers which act as bins,separated by commas and NO spaces
        clusteringBool = True
        
except:
    clustering = True
    clusteringBool= True

try:
    partitions=int(sys.argv[8])
except:
    partitions = 1
try:
    WSClean_grid = bool(sys.argv[9])
    if (WSClean_grid == True):
        ms_fieldcenter = False
    else: 
        ms_fieldcenter = True
except:
    WSClean_grid=False
    ms_fieldcenter=True


try:
    wsclean_path = sys.argv[10]
except:
    if (WSClean_grid==True):
        raise(SyntaxError("If WSClean Grid is set to True then path to wsclean fits file must be provided!"))
    else:
        print ("WSClean fits file not provided.")
        #wsclean_path= "/scratch/izar/krishna/MWA/WSClean/Simulation/simulation_MWA_Obsparams.ms_WSClean-dirty.fits"
        #wsclean_path = "/home/krishna/OSKAR/Example/simulation_oskarImaged_MWA_Obsparams_I.fits"
        wsclean_path= "/scratch/izar/krishna/MWA/WSClean/1133149192-187-188_Sun_10s_cal.ms_WSClean-dirty.fits"



print (f"Telescope Name:{telescope_name}")
print (f"MS file:{ms_file}")
print (f"Output Name:{outName}")
print (f"N_Pix:{N_pix} pixels")
print (f"FoV:{np.rad2deg(FoV)} deg")
print (f"N_level:{N_level} levels")
if (clusteringBool):
    print (f"Clustering Bool:{clusteringBool}")
    print (f"KMeans Clustering:{clustering}")
else:
    print(f"Clustering Bool:{clusteringBool}")
    print (f"Clustering:{clustering}")
print (f"WSClean_grid:{WSClean_grid}")
print (f"ms_fieldcenter:{ms_fieldcenter}")
print (f"WSClean Path: {wsclean_path}")
print (f"Partitions: {partitions}")



################################################################################################################################################################################
# Control Variables ########################################################################################
###########################################################################################################

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
#channel_id = np.array([4], dtype = np.int64)
#channel_id = np.array([4,5], dtype = np.int64)
channel_id = np.arange(64, dtype = np.int32)

# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bipp.Context("CPU")

filter_tuple = ['lsq','std'] # might need to make this a list

std_img_flag = True # put to true if std is passed as a filter

plotList= np.array([1,2])
# 1 is lsq
# 2 is levels
# 3 is WSClean
# 4 is WSClean v/s lsq comparison

# 1 2 and 3 are on the same figure - we can remove 2 and 3 from this figure also
# 4 is on a different figure with 1 and 2

#######################################################################################################################################################
# Observation set up ########################################################################################
#######################################################################################################################################################

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

wl = constants.speed_of_light / frequency.to_value(u.Hz)
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
opt.set_local_image_partition(bipp.Partition.grid([partitions,partitions,1]))
opt.set_local_uvw_partition(bipp.Partition.grid([partitions,partitions,1]))
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

num_time_steps = 0
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=1, ctx=ctx)
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
Eigs, N_eig, intensity_intervals = I_est.infer_parameters(return_eigenvalues=True)




if (clusteringBool == False):
    intensity_intervals=clustering # N_eig still to be obtained from parameter estimator????? IMP # 26 for 083 084 39 for????

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
    precision
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
    print (uvw.shape)

    if np.allclose(S.data, np.zeros(S.data.shape)):
        continue
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

lsq_levels = I_lsq_eq.data  # Nlevel, Npix, Npix
lsq_image = lsq_levels.sum(axis = 0)
print (f"Lsq Levels shape:{lsq_levels.shape}")

if (std_img_flag):
    std_levels = I_std_eq.data # Nlevel, Npix, Npix
    std_image = std_levels.sum(axis = 0)

    print (f"STD Levels shape:{std_levels.shape}")

#"""
# Code to output figure needed for BB paper 
if (telescope_name.lower() =="mwa"):
    if (ms_file== '/work/ska/MWA/simulation_MWA_Obsparams.ms/'):

        #SOLAR SIMULATION PAPER FIGURE
        path = "/scratch/izar/krishna/MWA/20151203_240MHz_psimas.sav"
        import scipy.io as io
        simulation = io.readsav(path)#, python_dict = True)

        trueImg = simulation['quantmap'][0][0][:, :]

        fig, ax = plt.subplots(2,4, figsize=(80, 40))

        simScale = ax[0, 0].imshow(trueImg, cmap="cubehelix") # Unit is K
        ax[0, 0].set_title("Simulation")
        ax[0, 0].axis('off')
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(simScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)


        WSClean_image = fits.getdata(wsclean_path)
        WSClean_image = WSClean_image.reshape(WSClean_image.shape[-2:])

        wscScale = ax[0, 1].imshow(WSClean_image, cmap="cubehelix") # Unit is K
        ax[0, 1].set_title("Dirty Image")
        ax[0, 1].axis('off')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(wscScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)


        lsqScale = ax[0, 2].imshow(np.fliplr(lsq_image), cmap="cubehelix")
        ax[0, 2].set_title("Bluebild Least-Squares Image")
        ax[0, 2].axis('off')
        divider = make_axes_locatable(ax[0, 2])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(lsqScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)

        residual_image = np.fliplr(lsq_image)-WSClean_image
        residual_norm = TwoSlopeNorm(vcenter=0, vmin=residual_image.min(), vmax=residual_image.max())
        residualScale= ax[0, 3].imshow(residual_image, cmap = "RdBu_r", norm=residual_norm)
        ax[0, 3].set_title("Residual Image")
        ax[0, 3].axis('off')
        divider = make_axes_locatable(ax[0, 3])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(residualScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)

        lsqScale = ax[1, 0].imshow(np.fliplr(lsq_levels[0, :, :]),cmap="cubehelix")
        ax[1, 0].set_title("Bluebild LSQ Level 0")
        ax[1, 0].axis('off')
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(lsqScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)

        lsqScale = ax[1, 1].imshow(np.fliplr(lsq_levels[1, :, :]), cmap="cubehelix")
        ax[1, 1].set_title("Bluebild LSQ Level 1")
        ax[1, 1].axis('off')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(lsqScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)


        lsqSCale = ax[1, 2].imshow(np.fliplr(lsq_levels[2, :, :]), cmap="cubehelix")
        ax[1, 2].set_title("Bluebild LSQ Level 2")
        ax[1, 2].axis('off')
        divider = make_axes_locatable(ax[1, 2])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(lsqScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)

        ax[1, 3].hist(np.log10(Eigs), bins=25)
        ax[1, 3].set_title("Eigenvalue Histogram")
        ax[1, 3].set_xlabel(r'$log_{10}(\lambda_{a})$')
        ax[1, 3].set_ylabel("Count")
        ax[1, 3].axvline(np.log10(2e9), color="r")
        ax[1, 3].axvline(np.log10(4e7), color="r")
    
    else: 
        # SOLAR OBSERVATION PAPER FIGURE
        fig, ax = plt.subplots(2,3, figsize=(60, 40))
        
        WSClean_image = fits.getdata(wsclean_path)
        WSClean_image = np.flipud(WSClean_image.reshape(WSClean_image.shape[-2:]))

        wscScale = ax[0, 0].imshow(WSClean_image, cmap="cubehelix") # Unit is K
        ax[0, 0].set_title("Dirty Image")
        ax[0, 0].axis('off')
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(wscScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)


        lsqScale = ax[0, 1].imshow(lsq_image, cmap="cubehelix")
        ax[0, 1].set_title("Bluebild Least-Squares Image")
        ax[0, 1].axis('off')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(lsqScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)

        residual_image = lsq_image-WSClean_image
        residual_norm = TwoSlopeNorm(vcenter=0, vmin=residual_image.min(), vmax=residual_image.max())
        residualScale= ax[0, 2].imshow(residual_image, cmap = "RdBu_r", norm=residual_norm)
        ax[0, 2].set_title("Residual Image")
        ax[0, 2].axis('off')
        divider = make_axes_locatable(ax[0, 2])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(residualScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)

        lsqScale = ax[1, 0].imshow(lsq_levels[0, :, :],cmap="cubehelix")
        ax[1, 0].set_title("Bluebild LSQ Level 0")
        ax[1, 0].axis('off')
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(lsqScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)

        lsqScale = ax[1, 1].imshow(lsq_levels[1, :, :], cmap="cubehelix")
        ax[1, 1].set_title("Bluebild LSQ Level 1")
        ax[1, 1].axis('off')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(lsqScale, cax)
        cbar.set_label('Flux (K)', rotation=270, labelpad=40)

        ax[1, 2].hist(np.log10(Eigs), bins=25)
        ax[1, 2].set_title("Eigenvalue Histogram")
        ax[1, 2].set_xlabel(r'$log_{10}(\lambda_{a})$')
        ax[1, 2].set_ylabel("Count")
        
        eigenvalue_binEdges = np.sort(np.unique(np.array(intensity_intervals))) [1:-1]  # select all but first and last bin edge (0 and 3e34)

        for eigenvalue_binEdge in eigenvalue_binEdges:
            ax[1, 2].axvline(np.log10(eigenvalue_binEdge), color="r")


elif (telescope_name.lower()=='lofar'):
    fig, ax = plt.subplots(1,5, figsize=(100, 20))

    lsqScale = ax[0].imshow(lsq_image, cmap="cubehelix")
    ax[0].set_title("BB LSQ")
    ax[0].axis('off')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)

    lsqScale = ax[1].imshow(lsq_levels[0, :, :], cmap="cubehelix")
    ax[1].set_title("LSQ Lvl 0")
    ax[1].axis('off')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)

    lsqScale = ax[2].imshow(lsq_levels[1, :, :], cmap="cubehelix")
    ax[2].set_title("LSQ Lvl 1")
    ax[2].axis('off')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)

    lsqScale = ax[3].imshow(lsq_levels[2, :, :], cmap="cubehelix")
    ax[3].set_title("LSQ Lvl 2")
    ax[3].axis('off')
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)

    lsqScale = ax[4].imshow(lsq_levels[3, :, :], cmap="cubehelix")
    ax[4].set_title("LSQ Lvl 3")
    ax[4].axis('off')
    divider = make_axes_locatable(ax[4])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)

elif (telescope_name.lower() == "skalow"):

    fig, ax = plt.subplots(2,3, figsize=(40, 20))

    lsqScale = ax[0, 0].imshow(lsq_image, cmap="cubehelix")
    ax[0, 0].set_title("BB LSQ")
    ax[0, 0].axis('off')
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)
    cbar.set_label('Flux (Jy/Beam)', rotation=270, labelpad=40)

    lsqScale = ax[0, 1].imshow(lsq_levels[0, :, :], cmap="cubehelix")
    ax[0, 1].set_title("LSQ Lvl 0")
    ax[0, 1].axis('off')
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)
    cbar.set_label('Flux (Jy/Beam)', rotation=270, labelpad=40)

    lsqScale = ax[0, 2].imshow(lsq_levels[1, :, :], cmap="cubehelix")
    ax[0, 2].set_title("LSQ Lvl 1")
    ax[0, 2].axis('off')
    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)
    cbar.set_label('Flux (Jy/Beam)', rotation=270, labelpad=40)

    lsqScale = ax[1, 0].imshow(lsq_levels[2, :, :], cmap="cubehelix")
    ax[1, 0].set_title("LSQ Lvl 2")
    ax[1, 0].axis('off')
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)
    cbar.set_label('Flux (Jy/Beam)', rotation=270, labelpad=40)

    lsqScale = ax[1, 1].imshow(lsq_levels[3, :, :], cmap="cubehelix")
    ax[1, 1].set_title("LSQ Lvl 3")
    ax[1, 1].axis('off')
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)
    cbar.set_label('Flux (Jy/Beam)', rotation=270, labelpad=40)
    
    lsqScale = ax[1, 2].imshow(lsq_levels[4, :, :], cmap="cubehelix")
    ax[1, 2].set_title("LSQ Lvl 4")
    ax[1, 2].axis('off')
    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)
    cbar.set_label('Flux (Jy/Beam)', rotation=270, labelpad=40)

fig.tight_layout()

fig.savefig(f"{outName}.pdf")
fig.savefig(f"{outName}.png")
"""


if (3 in plotList or 4 in plotList):
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
    ax_out[0, 0].axis('off')
    divider = make_axes_locatable(ax_out[0, 0])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(BBScale, cax)
    cbar.formatter.set_powerlimits((0, 0))

    # Output STD Image
    if (std_img_flag):
        BBScale = ax_out[1, 0].imshow(std_image, cmap = "cubehelix")
        ax_out[1, 0].set_title(r"STD IMG")
        ax_out[1, 0].axis('off')
        divider = make_axes_locatable(ax_out[1, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(BBScale, cax)
        cbar.formatter.set_powerlimits((0, 0))

    # output eigen levels
    if ((2 in plotList) or (3 in plotList)):
        for i in np.arange(eigenlevels):
            print (f"Loop {i}: vmin:{lsq_levels[i, :, :].max() * std_levels[i, :, :].min()/std_levels[i, :, :].max() if std_img_flag else lsq_levels[i, :, :].min()}")
            print (f"lsq_levels {i}.max():{lsq_levels[i, :, :].max()} std_levels {i}.min():{std_levels[i, :, :].min()} std_levels {i}.max():{std_levels[i, :, :].max()}")

            lsqScale = ax_out[0, i + 1].imshow(lsq_levels[i, :, :], cmap = "cubehelix", \
                        vmin = (lsq_levels[i, :, :].max() * std_levels[i, :, :].min()/std_levels[i, :, :].max() if std_img_flag else lsq_levels[i, :, :].min()))
            ax_out[0, i + 1].set_title(f"Lsq Lvl {i}")
            ax_out[0, i + 1].axis('off')
            divider = make_axes_locatable(ax_out[0, i + 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(lsqScale, cax)
            cbar.formatter.set_powerlimits((0, 0))

            if (std_img_flag):
                stdScale = ax_out[1, i + 1].imshow(std_levels[i, :, :], cmap = "cubehelix")
                ax_out[1, i + 1].set_title(f"Std Lvl {i}")
                ax_out[1, i + 1].axis('off')
                divider = make_axes_locatable(ax_out[1, i + 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(stdScale, cax)
                cbar.formatter.set_powerlimits((0, 0))

        # WSC Image
    if ((3 in plotList)):
        WSCleanScale = ax_out[0, -1].imshow(WSClean_image, cmap = "cubehelix")
        ax_out[0, -1].set_title(f"WSC IMG")
        ax_out[0, -1].axis('off')
        divider = make_axes_locatable(ax_out[0, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(WSCleanScale, cax)
        cbar.formatter.set_powerlimits((0, 0))
    
        if (std_img_flag):
            WSCleanScale = ax_out[1, -1].imshow(WSClean_image, cmap = "cubehelix")
            ax_out[1, -1].set_title(f"WSC IMG")
            ax_out[1, -1].axis('off')
            divider = make_axes_locatable(ax_out[1, -1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(WSCleanScale, cax)
            cbar.formatter.set_powerlimits((0, 0))

    if (2 not in plotList):
        fig_out.savefig(f"{outName}")
    else:
        fig_out.savefig(f"{outName}_lvls")
        
    print(f"{outName}.png saved.")

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
    print (f"{outName}_comparison.png saved.")
#"""

print(f'Elapsed time: {tt.time() - start_time} seconds.')