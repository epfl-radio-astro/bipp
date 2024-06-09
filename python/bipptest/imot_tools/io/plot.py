# ############################################################################
# plot.py
# =======
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# ############################################################################

"""
`Matplotlib <https://matplotlib.org/>`_ helpers.
"""

import pathlib

import matplotlib.axes as axes
import matplotlib.colors as colors
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as ax_grid
import pandas as pd
import pkg_resources as pkg

import bipp.imot_tools.util.argcheck as chk


@chk.check(dict(name=chk.is_instance(str), N=chk.allow_None(chk.is_integer)))
def cmap(name, N=None):
    """
    Load a custom colormap.

    All maps are defined under ``<ImoT_tools_dir>/data/io/colormap/``.

    Parameters
    ----------
    name : str
        Colormap name.
    N : int, optional
        Number of color levels. (Default: all).

        If `N` is smaller than the number of levels available in the colormap, then the last `N`
        colors will be used.

    Returns
    -------
    colormap : :py:class:`~matplotlib.colors.ListedColormap`

    Examples
    --------
    .. doctest::

       import numpy as np
       import matplotlib.pyplot as plt

       from imot_tools.io.plot import cmap

       x, y = np.ogrid[-1:1:100j, -1:1:100j]

       fig, ax = plt.subplots(ncols=2)
       ax[0].imshow(x + y, cmap='jet')
       ax[0].set_title('jet')

       cmap_name = 'jet_alt'
       ax[1].imshow(x + y, cmap=cmap(cmap_name))
       ax[1].set_title(cmap_name)

       fig.show()

    .. image:: _img/cmap_example.png
    """
    if (N is not None) and (N <= 0):
        raise ValueError("Parameter[N] must be a positive integer.")

    cmap_rel_dir = pathlib.Path("data", "io", "colormap")
    cmap_rel_path = cmap_rel_dir / f"{name}.csv"

    resource = "imot_tools"
    if pkg.resource_exists(resource, str(cmap_rel_path)):
        cmap_abs_path = pkg.resource_filename(resource, str(cmap_rel_path))
        col = pd.read_csv(cmap_abs_path).loc[:, ["R", "G", "B"]].values

        N = len(col) if (N is None) else N
        colormap = colors.ListedColormap(col[-N:])
        return colormap
    else:  # no cmap under that name.
        # List available cmaps.
        cmap_names = sorted(
            [
                pathlib.Path(_).stem
                for _ in pkg.resource_listdir(resource, str(cmap_rel_dir))
                if _.endswith("csv")
            ]
        )
        raise ValueError(
            f"No colormap named '{name}' available. Valid maps: {cmap_names}"
        )


@chk.check(dict(scm=chk.is_instance(cm.ScalarMappable), ax=chk.is_instance(axes.Axes)))
def colorbar(scm, ax):
    """
    Attach colorbar to side of a plot.

    Parameters
    ----------
    scm : :py:class:`~matplotlib.cm.ScalarMappable`
        Intensity scale.
    ax : :py:class:`~matplotlib.axes.Axes`
        Plot next to which the colorbar is placed.

    Returns
    -------
    cbar : :py:class:`~matplotlib.colorbar.Colorbar`

    Examples
    --------
    .. doctest::

       import matplotlib.pyplot as plt
       import numpy as np

       from imot_tools.io.plot import colorbar

       x, y = np.ogrid[-1:1:100j, -1:1:100j]

       fig, ax = plt.subplots()
       im = ax.imshow(x + y, cmap='jet')
       cb = colorbar(im, ax)

       fig.show()

    .. image:: _img/colorbar_example.png
    """
    fig = ax.get_figure()
    divider = ax_grid.make_axes_locatable(ax)
    ax_colorbar = divider.append_axes(
        "right", size="5%", pad=0.05, axes_class=axes.Axes
    )
    cbar = fig.colorbar(scm, cax=ax_colorbar)
    return cbar
