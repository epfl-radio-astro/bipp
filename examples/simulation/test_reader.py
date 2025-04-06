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



class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)
#  print(json.dumps(mapping, cls=NumpyArrayEncoder))

N_pix = 350
#  N_pix = 10000
FoV = np.deg2rad(10)
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
lmn_grid, xyz_grid = frame.make_grids(N_pix, FoV, field_center)

N_level = 3
time_slice = 1
I_est = bb_pe.ParameterEstimator(N_level, sigma=0.95)


with bipp.DatasetFile.open("test.h5") as dataset, bipp.ImageFile.create("image.h5", lmn_grid.transpose()) as image:
    for idx in range(0, dataset.num_samples(), time_slice):
        I_est.collect(dataset.eig_val(idx))

    intervals = I_est.infer_parameters()

    selection = {}
    filters = ["lsq", "std"]

    for filter_name in filters:
        for level in range(intervals.shape[0]):
            fi = bipp.filter.Filter(filter_name, intervals[level, 0], intervals[level,1])
            tag = f"{filter_name}_level_{level}"
            level_selection = {}
            for idx in range(0, dataset.num_samples(), time_slice):
                level_selection[idx] = fi(dataset.eig_val(idx))
            selection[tag] = level_selection


    image.set_meta("grid_x", xyz_grid[0].flatten())
    image.set_meta("grid_y", xyz_grid[1].flatten())
    image.set_meta("grid_z", xyz_grid[2].flatten())

    #  with open('selection.json', 'w') as f:
    #      json.dump(selection, f, cls=NumpyArrayEncoder)

    print("synthesis")

    comm = bipp.communicator.world()
    opt = bipp.NufftSynthesisOptions()
    #  opt.set_local_image_partition(bipp.Partition.grid([1,1,1]))
    opt.set_local_uvw_partition(bipp.Partition.grid([1,1,1]))
    ctx = bipp.Context("AUTO", comm)
    bipp.image_synthesis(ctx, opt, dataset, selection, image)


#  bipp.image_synthesis(mapping)



