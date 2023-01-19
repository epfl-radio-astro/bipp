# #############################################################################
# fits.py
# =======
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Helper functions to interact with standard-compliant FITS files.
"""

import astropy.io.fits as fits
import astropy.wcs as awcs
import numpy as np

import bipp.imot_tools.math.sphere.transform as transform


def wcs(file_name, ext=0):
    r"""
    Extract World Coordinate System (WCS) from a FITS extension.

    Parameters
    ----------
    file_name : path-like
        FITS file.
    ext : int / string
        FITS extension to query. (PrimaryHDU=0, etc.)

    Returns
    -------
    WCS : :py:class:`~astropy.wcs.WCS`
    """
    with fits.open(
        file_name, mode="readonly", memmap=True, lazy_load_hdus=True
    ) as hdulist:
        WCS = awcs.WCS(hdulist[ext].header)
        return WCS


def pix_grid(WCS):
    r"""
    Extract pixel ICRS coordinates in :math:`\mathbb{S}^{2}`.

    Parameters
    ----------
    WCS : :py:class:`~astropy.wcs.WCS`
        (N_1, N_2) data (after removal non-celestial axes).

    Returns
    -------
    R : :py:class:`~numpy.ndarray`
        (3, N_2, N_1) Cartesian ICRS pixel coordinates.

    Notes
    -----
    Since the FITS standard uses FORTRAN conventions for index/array-ordering,
    `WCS` dimensions/information are reversed w.r.t the C-ordering conventions
    of NumPy.
    What does this mean? Given a 2D WCS of shape (N_1, N_2), `R` will have shape
    (3, N_2, N_1).
    """
    WCS = WCS.celestial

    if len(WCS.array_shape) != 2:
        raise NotImplementedError(
            "Coordinates can only be extracted from " "IMAGE-like extensions."
        )
    if (WCS.wcs.lngtyp != "RA") and (WCS.wcs.lattyp != "DEC"):
        raise NotImplementedError(
            "pix_grid() assumes coordinates are provided " "in RA/DEC format."
        )

    N_1, N_2 = WCS.array_shape
    idx_1 = np.arange(1, N_1 + 1).reshape(N_1, 1)
    idx_2 = np.arange(1, N_2 + 1).reshape(1, N_2)
    origin = 1  # We choose 1 since the WCS header came from a FITS file,
    # hence follows FORTRAN indexing conventions.

    d_1, d_2 = WCS.all_pix2world(idx_1, idx_2, origin)  # [deg]
    if WCS.wcs.lng == 0:
        lon, lat = d_1, d_2
    else:
        lat, lon = d_1, d_2

    lon, lat = map(np.deg2rad, [lon, lat])  # [rad]
    R = transform.eq2cart(1, lat, lon)
    return R
