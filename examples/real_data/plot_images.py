import argparse
import numpy as np
import bipp
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--image", type=str, required=True, help="Image file"
)

args = parser.parse_args()

def plot_images(args):
    with bipp.ImageDataFile.open(args.image) as image_file:
        tags = image_file.tags()
        tags.sort()


        for t in tags:
            plt.figure()
            ax = plt.gca()
            image = image_file.get(t)
            plt.imshow(image, cmap ='cubehelix')
            plt.title(t)
            plt.colorbar()
            plt.savefig(f"{t}.png", dpi=200)
