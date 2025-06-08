import argparse
from bipp.apps.create_dataset import create_dataset
from bipp.apps.create_image_prop import create_image_prop
from bipp.apps.create_selection import create_selection
from bipp.apps.plot_images import plot_images
from bipp.apps.image_synthesis import image_synthesis

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(title='subcommands',
                                   description='valid subcommands',
                                   help='additional help',required=True)


#####################
# Create dataset args
#####################
description = "Convert dataset"

dataset_parser = subparsers.add_parser('dataset', description=description)

# Telescope Name
dataset_parser.add_argument(
    "-t",
    "--telescope",
    type=str,
    required=True,
    choices=["SKAlow", "LOFAR", "MWA"],
    help="Name of telescope.\n Currently SKALow, LOFAR and MWA accepted.",
)
# Path to MS File
dataset_parser.add_argument(
    "-ms", "--ms_file", type=str, required=True, help="Path to .ms file."
)
# Output filename
dataset_parser.add_argument(
    "-o", "--output", type=str, required=True, help="Name of output file."
)

dataset_parser.add_argument(
    "-r",
    "--range",
    nargs=3,
    type=int,
    help="Timestep index (start, end, step) to include in analysis. Give 3 integers: start end(exclusive) step.",
)

dataset_parser.add_argument(
    "-c",
    "--channel",
    nargs=2,
    type=int,
    help="Channels to include in analysis. Give 2 integers separated: start end(exclusive).",
)

dataset_parser.add_argument(
    "-d",
    "--data",
    type=str,
    default="DATA",
    help="Which column from the measurement set file to use. Eg: DATA, CORRECTED_DATA, MODEL_DATA",
)

dataset_parser.add_argument("-a", "--antenna", type=int, help="Number of antenna.")
dataset_parser.add_argument("-s", "--station", type=int, help="Number of stations.")

dataset_parser.set_defaults(func=create_dataset)


########################
# Create image prop args
########################

description = "Create image properties"

image_prop_parser = subparsers.add_parser('image_prop', description=description)

image_prop_parser.add_argument(
    "-d", "--dataset", type=str, required=True, help="datset file"
)

image_prop_parser.add_argument(
    "-f", "--fov", type=float, required=True, help="fov"
)

image_prop_parser.add_argument(
    "-w", "--width", type=int, required=True, help="image width"
)

image_prop_parser.add_argument(
    "-o", "--output", type=str, required=True, help="Name of output file."
)
image_prop_parser.set_defaults(func=create_image_prop)

#######################
# Create selection args
#######################

description = """
Create eigenvalue selections from a dataset. A selection is descripted through a 6-value tuple consisting of filter, number of levels, sigma value [0, 1.0], cluster function, minimum and maximum.

Example for creating a selection with 5 levels for eigenvalues in [0,inf) using the 95% smallest eigenvalues with log function for clustering, and one level containing all negative eigenvalues:

   -s lsq,5,0.95,log,0,inf -s lsq,1,1.0,none,-inf,0 -d dataset.h5 -o selection.json
"""

selection_parser = subparsers.add_parser('selection', description=description)

selection_parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset")

selection_parser.add_argument(
    "-s",
    "--selection",
    type=str,
    required=True,
    action="append",
    help="Selection description",
)

selection_parser.add_argument(
    "-r",
    "--range",
    nargs=3,
    type=int,
    help="Index range (start, end, step) to include. Provide 3 integers: start end(exclusive) step.",
)

selection_parser.add_argument(
    "-o", "--output", type=str, required=True, help="Name of output json file."
)

selection_parser.set_defaults(func=create_selection)


###########
# Synthesis
###########

description = "Compute image synthesis"

synthesis_parser = subparsers.add_parser('synthesis', description=description)

synthesis_parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset (hdf5)")

synthesis_parser.add_argument("-s", "--selection", type=str, required=True, help="Selection (json)")

synthesis_parser.add_argument("-i", "--image", type=str, required=True, help="Image properties (hdf5)")

synthesis_parser.add_argument("-o", "--output", type=str, required=True, help="Output image (hdf5)")

synthesis_parser.add_argument("-p", "--proc", type=str, required=False, default="auto", help="Processing unit")

synthesis_parser.add_argument("-f", "--float_precision", type=str, required=False, default="single", help="Floating point precision")

synthesis_parser.add_argument("-t", "--tol", type=float, required=False, default=0.001, help="NUFFT tolerance")

synthesis_parser.add_argument("--uvw_part", nargs=3, type=int, required=False, help="Processing unit")

synthesis_parser.set_defaults(func=image_synthesis)


###########
# Plotting
###########

description = "Plot images"

plot_parser = subparsers.add_parser('plot', description=description)

plot_parser.add_argument("-i", "--image", type=str, required=True, help="Image file (hdf5)")

plot_parser.set_defaults(func=plot_images)


def run_cli():
   args = parser.parse_args()
   args.func(args)


################
# Run Subcommand
################
if __name__ == "__main__":
   run_cli()

