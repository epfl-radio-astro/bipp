# #############################################################################
# source.py
# =========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Sky model generation.
"""

import collections.abc as abc
import pathlib
import shutil
import urllib.request

import astropy.coordinates as coord
import astropy.units as u
import bipp.imot_tools.math.sphere.transform as transform
import bipp.imot_tools.util.argcheck as chk
import numpy as np
import pandas as pd


def is_source_config(x):
    """
    Return :py:obj:`True` if `x` is a valid
    :py:class:`~bipp.phased_array.data_gen.source.SkyEmission` config specification.

    A source config spec is considered valid if it is a collection of pairs (a, b), with

    * a : :py:class:`~astropy.coordinates.SkyCoord`
        source position in the sky;
    * b : float
        source intensity.
    """
    if chk.is_instance(abc.Collection)(x):
        for entry in x:
            if not chk.is_instance(abc.Sequence)(entry):
                return False

            N_field = len(entry)
            if N_field != 2:
                return False

            intensity = entry[1]
            if not (chk.is_real(intensity) and (intensity > 0)):
                return False

            direction = entry[0]
            if not chk.is_instance(coord.SkyCoord)(direction):
                return False
        return True

    return False


class SkyEmission:
    """
    Container for storing position/intensity information of far-field sources.

    Args
        source_config
            Source configuration.
            Must satisfy :py:func:`~bipp.phased_array.data_gen.source.is_source_config`.
    """

    def __init__(self, source_config):
        N_src = len(source_config)
        intensity = [None] * N_src
        direction = [None] * N_src
        for i, (s_dir, s_int) in enumerate(source_config):
            intensity[i] = s_int
            direction[i] = s_dir.transform_to("icrs").cartesian.xyz.value

        self.__xyz = np.stack(direction, axis=0)
        self.__intensity = np.array(intensity)

    @property
    def xyz(self):
        """
        Returns
            :py:class:`~numpy.ndarray`
                (N_src, 3) Cartesian ICRS source locations.
        """
        return self.__xyz

    @property
    def intensity(self):
        """
        Returns
            :py:class:`~numpy.ndarray`
                (N_src,) source intensities.
        """
        return self.__intensity


def from_tgss_catalog(direction, FoV, N_src):
    """
    Generate :py:class:`~bipp.phased_array.data_gen.source.SkyEmission` from the
    `TGSS <http://tgssadr.strw.leidenuniv.nl/doku.php>`_ catalog.

    This function will automatically download and cache the catalog on disk for subsequent calls.

    Args
        direction : :py:class:`~astropy.coordinates.SkyCoord`
            Direction in the sky.
        FoV : float
            Spherical angle [rad] of the sky centered at `direction` from which sources are extracted.
        N_src : int
            Number of dominant sources to extract.

    Returns
        :py:class:`~bipp.phased_array.data_gen.source.SkyEmission`
            Sky model.
    """
    if FoV <= 0:
        raise ValueError("Parameter[FoV] must be positive.")

    if N_src <= 0:
        raise ValueError("Parameter[N_src] must be positive.")

    disk_path = (
        pathlib.Path.home() / ".bipp" / "catalog" / "TGSSADR1_7sigma_catalog.tsv"
    )

    if not disk_path.exists():
        # Download catalog from web.
        catalog_dir = disk_path.parent
        if not catalog_dir.exists():
            catalog_dir.mkdir(parents=True)

        web_path = (
            "http://tgssadr.strw.leidenuniv.nl/catalogs/TGSSADR1_7sigma_catalog.tsv"
        )
        print(f"Downloading catalog from {web_path}")
        with urllib.request.urlopen(web_path) as response:
            with disk_path.open(mode="wb") as f:
                shutil.copyfileobj(response, f)

    # Read catalog from disk path
    catalog_full = pd.read_csv(disk_path, sep="\t")

    lat = np.deg2rad(catalog_full.loc[:, "DEC"].values)
    lon = np.deg2rad(catalog_full.loc[:, "RA"].values)
    xyz = transform.eq2cart(1, lat, lon)
    I = catalog_full.loc[:, "Total_flux"].values * 1e-3  # mJy in catalog.

    # Reduce catalog to area of interest
    f_dir = direction.transform_to("icrs").cartesian.xyz.value
    mask = (f_dir @ xyz) >= np.cos(FoV / 2)

    if mask.sum() < N_src:
        raise ValueError("There are less than Parameter[N_src] sources in the field.")

    I_region, xyz_region = I[mask], xyz[:, mask]
    idx = np.argsort(I_region)[-N_src:]
    I_region, xyz_region = I_region[idx], xyz_region[:, idx]
    _, lat_region, lon_region = transform.cart2eq(*xyz_region)

    source_config = [
        (coord.SkyCoord(az * u.rad, el * u.rad, frame="icrs"), intensity)
        for el, az, intensity in zip(lat_region, lon_region, I_region)
    ]
    sky_model = SkyEmission(source_config)
    return sky_model
