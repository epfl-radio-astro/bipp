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



N_station = 24
N_pix = 1024
FoV = np.deg2rad(10)
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
lmn_grid, xyz_grid = frame.make_grids(N_pix, FoV, field_center)
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=40)

images = []
with bipp.ImageReader("image.h5") as reader:
    tags = reader.tags()

    for t in tags:
        images.append(reader.read(tags[1]).reshape(N_pix, N_pix))


images = np.array(images)
plt.imshow(images[0])
plt.savefig("test.png")
exit(0)

#  lsq_image = fi.get_filter_images("lsq", images)
#  std_image = fi.get_filter_images("std", images)

print(xyz_grid.shape)
print(images.shape)
print(xyz_grid[0])
I_lsq_eq = s2image.Image(images, xyz_grid)
#  I_std_eq = s2image.Image(std_image, xyz_grid)

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

#  plt.figure()
#  ax = plt.gca()
#  I_std_eq.draw(
#      catalog=sky_model.xyz.T,
#      ax=ax,
#      data_kwargs=dict(cmap="cubehelix"),
#      show_gridlines=False,
#      catalog_kwargs=dict(s=30, linewidths=0.5, alpha=0.5),
#  )
#  ax.set_title(
#      f"Bipp STD, sensitivity-corrected image (NUFFT)\n"
#      f"Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.\n"
#      f"Run time {np.floor(t2 - t1)} seconds."
#  )

#  plt.savefig("nufft_synthesis_std.png")
#  plt.figure()
#  titles = ["Strong sources", "Mild sources", "Faint Sources"]
#  for i in range(lsq_image.shape[0]):
#      plt.subplot(1, N_level, i + 1)
#      ax = plt.gca()
#      plt.title(titles[i])
#      I_lsq_eq.draw(
#          index=i,
#          catalog=sky_model.xyz.T,
#          ax=ax,
#          data_kwargs=dict(cmap="cubehelix"),
#          catalog_kwargs=dict(s=30, linewidths=0.5, alpha=0.5),
#          show_gridlines=False,
#      )

#  plt.suptitle(f"Bipp Eigenmaps")
#  plt.savefig("nufft_synthesis_lsq.png")
#  plt.show()
