"""
Convert MS files to BIPP dataset
"""

import argparse
from tqdm import tqdm as ProgressBar
import astropy.units as u
import numpy as np
import scipy.constants as constants
import bipp
import bipp.frame as frame
import bipp.measurement_set as measurement_set
import time as tt


parser = argparse.ArgumentParser()

# Telescope Name
parser.add_argument(
    "-t",
    "--telescope",
    type=str,
    required=True,
    choices=["SKAlow", "LOFAR", "MWA"],
    help="Name of telescope.\n Currently SKALow, LOFAR and MWA accepted.",
)
# Path to MS File
parser.add_argument(
    "-ms", "--ms_file", type=str, required=True, help="Path to .ms file."
)
# Output filename
parser.add_argument(
    "-o", "--output", type=str, required=True, help="Name of output file."
)

parser.add_argument(
    "-r",
    "--range",
    nargs=3,
    type=int,
    help="Timestep index (start, end, step) to include in analysis. Give 3 integers: start end(exclusive) step.",
)

parser.add_argument(
    "-c",
    "--channel",
    nargs=2,
    type=int,
    help="Channels to include in analysis. Give 2 integers separated: start end(exclusive).",
)

parser.add_argument(
    "-d",
    "--data",
    type=str,
    default="DATA",
    help="Which column from the measurement set file to use. Eg: DATA, CORRECTED_DATA, MODEL_DATA",
)

parser.add_argument("-a", "--antenna", type=int, help="Number of antenna.")
parser.add_argument("-s", "--station", type=int, help="Number of stations.")

# Number of Pixels
args = parser.parse_args()

if args.telescope.lower() == "skalow":
    ms = measurement_set.SKALowMeasurementSet(args.ms_file)
    N_antenna = 512 if args.antenna is None else args.antenna
elif args.telescope.lower() == "redundant":
    ms = measurement_set.GenericMeasurementSet(args.ms_file)
    N_antenna = ms.AntennaNumber() if args.antenna is None else args.antenna
elif args.telescope.lower() == "mwa":
    ms = measurement_set.MwaMeasurementSet(args.ms_file)
    N_antenna = 128 if args.antenna is None else args.antenna
elif args.telescope.lower() == "lofar":
    N_antenna = 37 if args.antenna is None else args.antenna
    ms = measurement_set.LofarMeasurementSet(
        args.ms_file, N_station=N_station, station_only=True
    )
else:
    raise (
        NotImplementedError(
            "A measurement set class for the telescope you are searching for has not been implemented yet - please feel free to implement the class yourself!"
        )
    )

field_center = ms.field_center

N_station = N_antenna if args.station is None else args.station

if args.range == None:
    timeStart = 0
    timeEnd = -1
    timeStep = 1
else:
    timeStart = args.range[0]
    timeEnd = args.range[1]
    timeStep = args.range[2]


if args.channel == None:
    channelStart = 0
    channelEnd = -1
    nChannel = ms.channels["FREQUENCY"].shape[0]
else:
    channelStart = args.channel[0]
    channelEnd = args.channel[0]
    nChannel = channelEnd - channelStart

if channelEnd - channelStart == 1:
    print("Single channel mode.")
    channel_id = np.arange(channelStart, channelEnd, dtype=np.int32)
else:
    channel_id = np.arange(channelStart, nChannel, dtype=np.int32)
    print(f"Multi-channel mode with {channelEnd - channelStart} channels.")


dec_deg = (field_center.frame.dec * u.deg).value
ra_deg = (field_center.frame.ra * u.deg).value
print(f"N_station: {N_station} , N_antenna: {N_antenna}")
print(f"ra (deg): {ra_deg} , dec (deg): {dec_deg}")

n_written = 0
with bipp.DatasetFile.create(
    args.output, args.telescope.lower(), N_antenna, N_station, ra_deg, dec_deg
) as dataset:
    for t, f, S in ProgressBar(
        ms.visibilities(
            channel_id=channel_id,
            time_id=slice(timeStart, timeEnd, timeStep),
            column=args.data,
        )
    ):
        wl = constants.speed_of_light / f.to_value(u.Hz)
        XYZ = ms.instrument(t)
        W = ms.beamformer(XYZ, wl)
        #  S, W = measurement_set.filter_data(S, W)

        UVW_baselines_t = ms.instrument.baselines(
            t, uvw=True, field_center=ms.field_center
        )
        uvw = frame.reshape_uvw(UVW_baselines_t)

        if np.allclose(S.data, np.zeros(S.data.shape)):
            continue

        v, d, scale = bipp.eigh_gram(wl, S.data, W.data, XYZ.data)
        dataset.write(t.value, wl, scale, v, d, uvw)
        n_written += 1

if n_written:
    print(f"{n_written} samples exported")
else:
    print("WARNING: no samples exported!\n")

