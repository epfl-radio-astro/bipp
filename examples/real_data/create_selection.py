"""
Script using pythons argparse to run bipp on real data sets. 
"""

import argparse
from tqdm import tqdm as ProgressBar
import astropy.units as u
import numpy as np
import scipy.constants as constants
import bipp
import time as tt
import bipp.parameter_estimator as bipp_pe
import bipp.selection as bipp_se


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, required=True, 
                    help="Dataset")

parser.add_argument("-s", "--selection", type=str, required=True, action='append',
                    help="Selection description")

parser.add_argument("-r","--range", nargs=3, type=int,
                    help="Index range (start, end, step) to include. Provide 3 integers: start end(exclusive) step.")

parser.add_argument("-o","--output", type=str, required=True,
                    help="Name of output json file.")

args = parser.parse_args()


# std,5,0.95,log,0,inf


with bipp.DatasetFile.open(args.dataset) as dataset:
    if (args.range==None):
        indexStart = 0
        indexEnd = dataset.num_samples()
        indexStep = 1
    else:
        indexStart = args.range[0]
        indexEnd = args.range[1]
        indexStep = args.range[2]

    # collect eigenvalues
    eig_values = []
    for idx in range(indexStart, indexEnd, indexStep):
        eig_values.append(dataset.eig_val(idx))

    s_dict = {}
    for s_string in args.selection:
        s = s_string.split(',')
        if len(s) != 6:
            raise ValueError(f"Expected comma separated list if size 6. Input: {s_string}")
        f = s[0].lower()
        n_cluster = int(s[1])
        sigma = float(s[2])
        cluster_func = s[3].lower()
        d_min = float(s[4])
        d_max = float(s[5])

        intervals = bipp_pe.infer_intervals(n_cluster, sigma, cluster_func, d_min, d_max, eig_values)

        for level in range(intervals.shape[0]):
            fi = bipp.filter.Filter(f, intervals[level, 0], intervals[level, 1])
            level_selection = {}
            for idx, ev in enumerate(eig_values):
                d = np.array(ev)
                level_selection[idx] = fi(d)
            tag = f"{s_string}_level_{level}"
            s_dict[tag] = level_selection

    bipp_se.export_selection(s_dict, args.output)

