"""
Script using pythons argparse to run bipp on real data sets. 

Outstanding questions:
What exactly does time slice do - do we require for it to be non 1 ????
"""

import argparse
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
#"Eg:\npython realData.py [telescope name(string)] [path_to_ms(string)] [output_name(string)] [N_pix(int)] [FoV(float(deg))] [N_levels(int)] [Clustering(bool/list_of_eigenvalue_bin_edges)] [partitions] [WSCleangrid(bool)] [WSCleanPath(string)]"))

start_time= tt.time()
################################################################################################################################################################################
############################################ INPUT ####################################################################################################################################
################################################################################################################################################################################

parser = argparse.ArgumentParser()

# Telescope Name
parser.add_argument("telescope", type=str,
                    help="Name of telescope.\n Currently SKALow, LOFAR and MWA accepted.")
# Path to MS File
parser.add_argument("ms_file", type=str,
                    help="Path to .ms file.")
# Output filename
parser.add_argument("-o","--output", type=str, 
                    help="Name of output file.")
# Number of Pixels
parser.add_argument("-n","--npix", type=int,
                    help="Number of pixels in output image.")
# Field of View
parser.add_argument("-f","--fov", type=float,
                    help="Field of View of image (in degrees).")
# Number of Energy Levels
parser.add_argument("-l", "--nlevel", type = int,
                    help="Number of Bluebild energy levels.")
# Clustering
parser.add_argument("-b", "--clustering",
                    help="Clustering (Boolean/ List of eigenvalue bin edges separated by commas)")
# Partitions
parser.add_argument("-p", "--partition", type = int,
                    help="Number of partitions for nuFFT.")
# Channels
parser.add_argument("-c", "--channel",
                    help="Channels to include in analysis. Give 2 integers separated by a comma: start,end(exclusive).")
# Time Steps
parser.add_argument("-t", "--timestep",
                    help="Timesteps to include in analysis. Give 2 integers separated by a comma: start,end(exclusive).")

parser.add_argument("-w", "--wsclean",
                    help="Path to WSClean .fits file (For comparison)")

parser.add_argument("-g", "--grid", type=str,
                    help="Which grid to use. Options are \n ms (grid defined by the Npix, FoV inputted earlier and field center defined by the ms file),\nwsclean (grid given by wsclean .fits file;requires path to wsclean file as well), and\n RA,Dec (Npix, FoV and field center all provided by user; in this case provide the user field center in RA and Dec)")

parser.add_argument("--column", type=str,
                    help="Which column from the measurement set file to image. Eg: DATA, CORRECTED_DATA, MODEL_DATA")

args = parser.parse_args()

if (args.telescope.lower()=="skalow"):
    ms=measurement_set.SKALowMeasurementSet(args.ms_file)
    N_station, N_antenna = 512, 512

elif (args.telescope.lower()=="mwa"):
    ms = measurement_set.MwaMeasurementSet(args.ms_file)
    N_station, N_antenna = 128, 128
    #N_station, N_antenna = 58, 58

elif (args.telescope.lower()=="lofar"):
    N_station, N_antenna = 37, 37 # netherlands, 52 international
    ms = measurement_set.LofarMeasurementSet(args.ms_file, N_station = N_station, station_only=True)

else: 
    raise(NotImplementedError("A measurement set class for the telescope you are searching for has not been implemented yet - please feel free to implement the class yourself!"))


if (args.output==None):
    args.output = "test"

if (args.npix==None):
    args.npix = 2000

if (args.fov==None):
    args.fov = np.deg2rad(7)
else:
    args.fov = np.deg2rad(args.fov)

if (args.nlevel==None):
    args.nlevel = 4

if (args.clustering==None):
    args.clustering=True
    clusteringBool=True
else:
    try:
        clusterEdges = np.array(args.clustering.split(","), dtype=np.float32)
        clusteringBool = False
        binStart = clusterEdges[0]
        clustering = []
        for binEdge in clusterEdges[1:]:
            binEnd = binEdge
            clustering.append([binEnd, binStart])
            binStart = binEnd
        clustering = np.asarray(clustering, dtype=np.float32)
    except:
        clustering = bool(args.clustering) # True or set of numbers which act as bins,separated by commas and NO spaces
        clusteringBool = True

if (args.partition==None):
    args.partition = 1

if (args.timestep==None):
    timeStart = 0
    timeEnd = -1
else:
    [timeStart, timeEnd] = np.array(args.timestep.split(","), dtype=np.int32)

if (args.channel==None):
    channelStart = 0
    channelEnd = -1
    nChannel = ms.channels["FREQUENCY"].shape[0]
else:
    [channelStart,channelEnd] = np.array(args.channel.split(","), dtype=np.int32)
    nChannel = channelEnd-channelStart

if (args.wsclean==None):
    args.wsclean="/work/ska/MWA/WSClean/1133149192-084-085_Sun_10s_cal.ms_WSClean-image.fits"  # add wsclean file path here

if (args.grid == None):
    ms_fieldcenter = True
    args.grid= "ms"
elif (args.grid.lower()=="ms"):
    ms_fieldcenter = True   
elif (args.grid.lower()=="wsclean"):
    ms_fieldcenter = False
elif (len(args.grid.split(",")) ==2):
    ms_fieldcenter = False
    [RA,Dec] = np.array(args.grid.split(","), dtype=np.float32)
else:
    raise NotImplementedError ("Only wsclean, ms and RA,Dec (degrees) grids have been defined so far.")

if (args.column==None):
    args.column = "DATA"
    

    
print (f"Telescope Name:{args.telescope}")
print (f"MS file:{args.ms_file}")
print (f"Output Name:{args.output}")
print (f"N_Pix:{args.npix} pixels")
print (f"FoV:{np.rad2deg(args.fov)} deg")
print (f"N_level:{args.nlevel} levels")
print(f"Clustering Bool:{clusteringBool}")
kmeans="kmeans"
print (f"Clustering:{kmeans if clusteringBool else args.clustering}")
print (f"grid:{args.grid}")
print (f"ms_fieldcenter:{ms_fieldcenter}")
print (f"ms channel start:{channelStart} channel end: {channelEnd}")
print (f"ms timestep start: {timeStart} timestep end: {timeEnd}")
print (f"WSClean Path:{args.wsclean}")
print (f"Partitions:{args.partition}")

##############################################################################################################
############################################ Control Variables ############################################
##############################################################################################################

# Column Name: Column in MS file to be imaged (DATA is usually uncalibrated, CORRECTED_DATA is calibrated and MODEL_DATA contains WSClean model output)
#args.column = "CORRECTED_DATA"

# IF USING WSCLEAN IMAGE GRID: sampling wrt WSClean grid
# 1 means the output will have same number of pixels as WSClean image
# N means the output will have WSClean Image/N pixels
sampling = 1

# error tolerance for FFT
eps = 1e-3

#precision of calculation
precision = 'single'

# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bipp.Context("AUTO")

filter_tuple = ['lsq','std'] # might need to make this a list

filter_negative_eigenvalues= True

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

if (channelEnd - channelStart == 1): 
    frequency = ms.channels["FREQUENCY"][channelStart:channelEnd]
    print ("Single channel mode.")
    channel_id = np.arange(channelStart, channelEnd, dtype = np.int32)
        
else:
    frequency = ms.channels["FREQUENCY"][channelStart] + (ms.channels["FREQUENCY"][channelEnd] - ms.channels["FREQUENCY"][channelStart])/2
    channel_id = np.arange(channelStart, nChannel, dtype = np.int32)
    print (f"Multi-channel mode with {channelEnd - channelStart}channels.")


wl = constants.speed_of_light / frequency.to_value(u.Hz)
print (f"wl:{wl}; f: {frequency}")

if (args.grid == "ms"):
    field_center = ms.field_center
    print ("Self generated grid used based on ms fieldcenter")
elif (args.grid =="wsclean"): 
    with fits.open(args.wsclean, mode="readonly", memmap=True, lazy_load_hdus=True) as hdulist:
        cl_WCS = awcs.WCS(hdulist[0].header)
        cl_WCS = cl_WCS.sub(['celestial'])
        cl_WCS = cl_WCS.slice((slice(None, None, sampling), slice(None, None, sampling)))

    field_center = ms.field_center

    width_px, height_px= 2*cl_WCS.wcs.crpix 
    cdelt_x, cdelt_y = cl_WCS.wcs.cdelt 
    FoV = np.deg2rad(abs(cdelt_x*width_px) )
    print ("WSClean Grid used.")
elif (len(args.grid.split(",")) == 2):
    field_center = coord.SkyCoord(ra=RA * u.deg, dec=Dec * u.deg, frame="icrs")
    print ("Self generated grid used based on user fieldcenter.")
else:
    raise NotImplementedError ("This gridstyle has not been implemented.")


lmn_grid, xyz_grid = frame.make_grids(args.npix, args.fov, field_center)

gram = bb_gr.GramBlock(ctx)

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
opt.set_local_image_partition(bipp.Partition.grid([args.partition,args.partition,1]))
opt.set_local_uvw_partition(bipp.Partition.grid([args.partition,args.partition,1]))

print("N_pix = ", args.npix)
print("precision = ", precision)
print("Proc = ", ctx.processing_unit)

print (f"Initial set up takes {tt.time() - start_time} s")

########################################################################################
### Intensity Field  Parameter Estimation ##############################################
########################################################################################
pe_t = tt.time()
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PARAMETER ESTIMATION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

num_time_steps = 0
I_est = bb_pe.IntensityFieldParameterEstimator(args.nlevel, sigma=1, ctx=ctx)
#for t, f, S, uvw_t in ProgressBar(
for t, f, S in ProgressBar(
        #ms.visibilities(channel_id=[channel_id], time_id=slice(timeStart, timeEnd, 1), column=args.column, return_UVW = True)
        ms.visibilities(channel_id=channel_id, time_id=slice(timeStart, timeEnd, 1), column=args.column)
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
print (f"Parameter Estimator takes: {tt.time() - pe_t} s")

########################################################################################
# Imaging ########################################################################################
########################################################################################
im_t = tt.time()

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
        ms.visibilities(channel_id=channel_id, time_id=slice(timeStart, timeEnd, 1), column=args.column)
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

lsq_image = imager.get("LSQ").reshape((-1, args.npix, args.npix))
if (filter_negative_eigenvalues):
    I_lsq_eq = s2image.Image(lsq_image.reshape(args.nlevel,lsq_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
else:
    I_lsq_eq = s2image.Image(lsq_image.reshape(args.nlevel + 1, lsq_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
print("lsq_image.shape =", lsq_image.shape)

if (std_img_flag):
    std_image = imager.get("STD").reshape((-1, args.npix, args.npix))
    if (filter_negative_eigenvalues):
        I_std_eq = s2image.Image(std_image.reshape(args.nlevel, std_image.shape[-2], lsq_image.shape[-1]), xyz_grid)
    else:
        I_std_eq = s2image.Image(std_image.reshape(args.nlevel + 1, std_image.shape[-2], std_image.shape[-1]), xyz_grid)
    print("std_image.shape =", std_image.shape)

print (f"Imaging takes :{tt.time() - im_t} s")

####################################################################################################################################
########################################## Plotting and output of .fits file #####################################################
####################################################################################################################################

pf_t = tt.time()

lsq_levels = I_lsq_eq.data  # Nlevel, Npix, Npix
hdu_lsqlvls = fits.ImageHDU(lsq_levels)

lsq_image = lsq_levels.sum(axis = 0)
hdu_lsq =fits.PrimaryHDU(lsq_image)

hdul = fits.HDUList([hdu_lsq, hdu_lsqlvls])

fig, ax = plt.subplots(1, args.nlevel+1)

if (std_img_flag):

    fig, ax = plt.subplots(2, args.nlevel+1)

    std_levels = I_std_eq.data  # Nlevel, Npix, Npix
    hdu_stdlvls = fits.ImageHDU(std_levels)

    std_image = std_levels.sum(axis = 0)
    hdu_std = fits.ImageHDU(std_image)
    
    hdul.append(hdu_std)
    hdul.append(hdu_stdlvls)

    # Plot Std Summed Image
    stdScale = ax[1,0].imshow(std_image)
    ax[1, 0].set_title("BB STD")
    ax[1, 0].axis('off')
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(stdScale, cax)
    cbar.set_label('Flux (Jy)', rotation=270, labelpad=40)

    # Plot Std Level Images  
    for i in np.arange(args.nlevel):
        stdScale = ax[1, i+1].imshow(std_levels[i, :, :])
        ax[1, i+1].set_title(f"STD {i}")
        ax[1, i+1].axis('off')
        divider = make_axes_locatable(ax[1, i+1])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cbar = plt.colorbar(stdScale, cax)
        cbar.set_label('Flux (Jy)', rotation=270, labelpad=40)

# Plot Lsq Summed Image    
lsqScale = ax[0, 0].imshow(lsq_image)
ax[0, 0].set_title("BB LSQ")
ax[0, 0].axis('off')
divider = make_axes_locatable(ax[0, 0])
cax = divider.append_axes("right", size = "5%", pad = 0.05)
cbar = plt.colorbar(lsqScale, cax)
cbar.set_label('Flux (Jy)', rotation=270, labelpad=40)

# Plot Lsq Level Image
for i in np.arange(args.nlevel):
    lsqScale = ax[0, i+1].imshow(lsq_levels[i, :, :])
    ax[0, i+1].set_title(f"LSQ {i}")
    ax[0, i+1].axis('off')
    divider = make_axes_locatable(ax[0, i+1])
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(lsqScale, cax)
    cbar.set_label('Flux (Jy)', rotation=270, labelpad=40)

fig.savefig(f"{args.output}.png")

hdul.writeto(f"{args.output}.fits", overwrite=True)

print (f"Plotting and fits output time:{tt.time() - pf_t} s")
print (f"Total time: {tt.time()- start_time} s")