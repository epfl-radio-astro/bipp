"""
Compute image synthesis
"""

import argparse
from argparse import RawTextHelpFormatter
import astropy.units as u
import numpy as np
import bipp
import json


def image_synthesis(args):
    with bipp.DatasetFile.open(args.dataset) as dataset, bipp.ImagePropFile.open(args.image) as image_prop, open(args.selection) as selection_file:
        selection = json.load(selection_file)
        # convert index strings to int
        selection = {tag:{int(i): values for i,values in samples.items()} for tag,samples in selection.items()}

        comm = bipp.communicator.world()
        ctx = bipp.Context(args.proc, comm)

        opt = bipp.NufftSynthesisOptions()
        opt.set_tolerance(args.tol)
        opt.set_precision(args.float_precision)
        if args.uvw_part is not None:
            opt.set_local_uvw_partition(bipp.Partition.grid(args.uvw_part))
        bipp.image_synthesis(ctx, opt, dataset, selection, image_prop, args.output)

