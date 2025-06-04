import argparse
import create_dataset from create_dataset
import create_image_prop from create_image_prop
import create_selection from create_selection
import plot_images from plot_images

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(title='subcommands',
                                   description='valid subcommands',
                                   help='additional help')

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


