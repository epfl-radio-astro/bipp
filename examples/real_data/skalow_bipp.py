import numpy as np, scipy.constants as constants, time as tt, matplotlib.pyplot as plt
from tqdm import tqdm as ProgressBar

import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime

import bipp
from bipp.imot_tools.io import s2image, fits
from bipp.imot_tools.io.plot import cmap
import bipp.beamforming as beamforming
import bipp.gram as bb_gr
import bipp.parameter_estimator as bb_pe
import bipp.source as source
import bipp.instrument as instrument
import bipp.frame as frame
import bipp.statistics as statistics

from measurement_set import SKALowMeasurementSet, filter_data #TODO: to substitued in the future

ctx = bipp.Context("AUTO")

time_slice = 1
N_station = 512
N_antenna = N_station
N_level = 10

# SKA-Low
path_out = '/scratch/snx3000/mibianco/test_bipp/'
path_img = '/project/c31/SKA_low_images/eos_fits/'
path_ms = '/project/c31/SKA_low_images/longobs_skalow/'

fname_img = 'EOS_21cm-gf_202MHz'
#fname_img = 'EOS_21cm_202MHz'
fname_prefix = fname_img+'_4h1d_1000'

fname_ms = path_ms+fname_prefix+'.MS'
data_column="DATA"  

ms = SKALowMeasurementSet(fname_ms)
channel_id = 0
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
dev = ms.instrument
mb = ms.beamformer

# Observation
FoV = np.deg2rad(1.02011e+01)
field_center = ms.field_center
time = ms.time['TIME'][:time_slice]
gram = bb_gr.GramBlock(ctx)

cl_WCS = fits.wcs(path_img+fname_img+'.fits').sub(['celestial']) 

# (3, N_cl_lon, N_cl_lat) ICRS reference frame
xyz_grid = fits.pix_grid(cl_WCS)
N_pix = xyz_grid.shape[1]

# (3, N_cl_lon, N_cl_lat) lmn frame
uvw_frame = frame.uvw_basis(field_center)
lmn_grid = np.tensordot(np.linalg.inv(uvw_frame), xyz_grid, axes=1).reshape(3, -1)

"""
N_pix = 1000
lmn_grid, xyz_grid = frame.make_grids(N_pix, FoV, field_center)
"""

### NUFFT Sythesis options ===========================================================
opt = bipp.NufftSynthesisOptions()
opt.set_tolerance(1e-3)
opt.set_collect_group_size(40) # lower == less memory == slow performance
opt.set_local_image_partition(bipp.Partition.grid([1,1,1]))
opt.set_local_uvw_partition(bipp.Partition.auto())
precision = "single"

print('''You are running bluebild on file: %s
         with the following input parameters:
         %d timesteps
         %d stations
         clustering into %d levels
         The output grid will be %dx%d = %d pixels''' %(fname_ms, len(time), N_station, N_level, xyz_grid.shape[1],  xyz_grid.shape[2],  lmn_grid.shape[1]))

### Intensity Field =================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=1., ctx=ctx)
for i_t, t in enumerate(ProgressBar(time[slice(None, None, 100)])):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    #S = vis(XYZ, W, wl)
    tobs, f, S = next(ms.visibilities(channel_id=[channel_id], time_id=slice(i_t, i_t+1, None), column=data_column))
    S, _ = filter_data(S, W)
    I_est.collect(S, G)

N_eig, intensity_intervals = I_est.infer_parameters()
# TODO : delete the context 
#ctx = None

# Imaging
imager = bipp.NufftSynthesis(ctx, opt, N_antenna, N_station, intensity_intervals.shape[0], ["LSQ", "SQRT"], lmn_grid[0], lmn_grid[1], lmn_grid[2], precision)

for i_t, t in enumerate(ProgressBar(time)):
    XYZ = dev(t)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    W = mb(XYZ, wl)
    #S = vis(XYZ, W, wl)

    tobs, f, S = next(ms.visibilities(channel_id=[channel_id], time_id=slice(i_t, i_t+1, None), column=data_column))
    S, _ = filter_data(S, W)

    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    imager.collect(N_eig, wl, intensity_intervals, W.data, XYZ.data, uvw, S.data)

lsq_image = imager.get("LSQ").reshape((-1, N_pix, N_pix))
sqrt_image = imager.get("SQRT").reshape((-1, N_pix, N_pix))
#"""
# TODO : store images LSQ and SQRT 
intI_lsq_eq = s2image.Image(lsq_image, xyz_grid)
intI_sqrt_eq = s2image.Image(sqrt_image, xyz_grid)
np.save('%sintI_lst_%s_%d.npy' %(path_out, fname_prefix, N_level), intI_lsq_eq.data)
np.save('%sintI_sqrt_%s_%d.npy' %(path_out, fname_prefix, N_level), intI_sqrt_eq.data)

# TODO : delete the context: imager = None, ctx = None, 
imager, ctx = None, None

# TODO : then reallocate the two above
ctx = bipp.Context("AUTO")
imager = bipp.NufftSynthesis(ctx, opt, N_antenna, N_station, intensity_intervals.shape[0], ["LSQ", "SQRT"], lmn_grid[0], lmn_grid[1], lmn_grid[2], precision)
#"""

### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=1., ctx=ctx)
for t in ProgressBar(time[slice(None, None, 100)]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)

N_eig = S_est.infer_parameters()

# Imaging
sensitivity_intervals = np.array([[0, np.finfo("f").max]])
imager = None  # release previous imager first to some additional memory
imager = bipp.NufftSynthesis(ctx, opt, N_antenna, N_station, sensitivity_intervals.shape[0], ["INV_SQ"], lmn_grid[0], lmn_grid[1], lmn_grid[2], precision)

for t in ProgressBar(time):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    imager.collect(N_eig, wl, sensitivity_intervals, W.data, XYZ.data, uvw, None)

sensitivity_image = imager.get("INV_SQ").reshape((-1, N_pix, N_pix))

# Save and Plot Results ================================================================
I_lsq_eq = s2image.Image(lsq_image / sensitivity_image, xyz_grid)
I_sqrt_eq = s2image.Image(sqrt_image / sensitivity_image, xyz_grid)

np.save('%setienne2I_lst_%s_%d.npy' %(path_out, fname_prefix, N_level), I_lsq_eq.data)
np.save('%setienne2I_sqrt_%s_%d.npy' %(path_out, fname_prefix, N_level), I_sqrt_eq.data)
"""
# Interpolate image to MS grid-frame for NUFFT
f_interp = (I_lsq_eq.data.reshape(N_level, N_pix, N_pix).transpose(0, 2, 1))
I_lsq_eq_interp = s2image.WCSImage(f_interp, cl_WCS)
I_lsq_eq_interp.to_fits('%sI_lst_%s_%d.fits' %(path_out, fname_prefix, N_level))
"""