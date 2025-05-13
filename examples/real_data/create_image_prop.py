"""
Convert MS files to BIPP dataset
"""

import argparse
import astropy.units as u
import numpy as np
import bipp
import astropy.coordinates as coord
import bipp.frame as frame

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--dataset", type=str, required=True, help="datset file"
)

parser.add_argument(
    "-f", "--fov", type=float, required=True, help="fov"
)

parser.add_argument(
    "-w", "--width", type=int, required=True, help="image width"
)

parser.add_argument(
    "-o", "--output", type=str, required=True, help="Name of output file."
)

# Number of Pixels
args = parser.parse_args()

with bipp.DatasetFile.open(args.dataset) as dataset:
    field_center = coord.SkyCoord(ra=dataset.ra_deg() * u.deg, dec=dataset.dec_deg() * u.deg, frame="icrs")

lmn_grid, xyz_grid = frame.make_grids(args.width, np.deg2rad(args.fov), field_center)

bipp.ImagePropFile.create(args.output, args.width, args.width, args.fov, lmn_grid.transpose())
