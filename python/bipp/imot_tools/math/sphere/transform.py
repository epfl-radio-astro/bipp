# ##############################################################################
# transform.py
# ============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# ##############################################################################

"""
Coordinate transforms.
"""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np

import bipp.imot_tools.util.argcheck as chk


@chk.check(
    dict(
        r=chk.accept_any(chk.is_real, chk.has_reals),
        colat=chk.accept_any(chk.is_real, chk.has_reals),
        lon=chk.accept_any(chk.is_real, chk.has_reals),
    )
)
def pol2eq(r, colat, lon):
    """
    Polar coordinates to Equatorial coordinates.

    Parameters
    ----------
    r : float or :py:class:`~numpy.ndarray`
        Radius.
    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].
    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius.

    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].

    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].
    """
    lat = (np.pi / 2) - colat
    return r, lat, lon


@chk.check(
    dict(
        r=chk.accept_any(chk.is_real, chk.has_reals),
        lat=chk.accept_any(chk.is_real, chk.has_reals),
        lon=chk.accept_any(chk.is_real, chk.has_reals),
    )
)
def eq2pol(r, lat, lon):
    """
    Equatorial coordinates to Polar coordinates.

    Parameters
    ----------
    r : float or :py:class:`~numpy.ndarray`
        Radius.
    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].
    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius.

    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].

    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].
    """
    colat = (np.pi / 2) - lat
    return r, colat, lon


@chk.check(
    dict(
        r=chk.accept_any(chk.is_real, chk.has_reals),
        lat=chk.accept_any(chk.is_real, chk.has_reals),
        lon=chk.accept_any(chk.is_real, chk.has_reals),
    )
)
def eq2cart(r, lat, lon):
    """
    Equatorial coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : float or :py:class:`~numpy.ndarray`
        Radius.
    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].
    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, ...) Cartesian XYZ coordinates.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.math.sphere.transform import eq2cart

    .. doctest::

       >>> xyz = eq2cart(1, 0, 0)
       >>> np.around(xyz, 2)
       array([[1.],
              [0.],
              [0.]])
    """
    r = np.array([r]) if chk.is_scalar(r) else np.array(r, copy=False)
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    XYZ = (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
        .to_cartesian()
        .xyz.to_value(u.dimensionless_unscaled)
    )
    return XYZ


@chk.check(
    dict(
        r=chk.accept_any(chk.is_real, chk.has_reals),
        colat=chk.accept_any(chk.is_real, chk.has_reals),
        lon=chk.accept_any(chk.is_real, chk.has_reals),
    )
)
def pol2cart(r, colat, lon):
    """
    Polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : float or :py:class:`~numpy.ndarray`
        Radius.
    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].
    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, ...) Cartesian XYZ coordinates.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.math.sphere.transform import pol2cart

    .. doctest::

       >>> xyz = pol2cart(1, 0, 0)
       >>> np.around(xyz, 2)
       array([[0.],
              [0.],
              [1.]])
    """
    lat = (np.pi / 2) - colat
    return eq2cart(r, lat, lon)


@chk.check(
    dict(
        x=chk.accept_any(chk.is_real, chk.has_reals),
        y=chk.accept_any(chk.is_real, chk.has_reals),
        z=chk.accept_any(chk.is_real, chk.has_reals),
    )
)
def cart2pol(x, y, z):
    """
    Cartesian coordinates to Polar coordinates.

    Parameters
    ----------
    x : float or :py:class:`~numpy.ndarray`
        X coordinate.
    y : float or :py:class:`~numpy.ndarray`
        Y coordinate.
    z : float or :py:class:`~numpy.ndarray`
        Z coordinate.

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius.

    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].

    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.math.sphere.transform import cart2pol

    .. doctest::

       >>> r, colat, lon = cart2pol(0, 1 / np.sqrt(2), 1 / np.sqrt(2))

       >>> np.around(r, 2)
       1.0

       >>> np.around(np.rad2deg(colat), 2)
       45.0

       >>> np.around(np.rad2deg(lon), 2)
       90.0
    """
    cart = coord.CartesianRepresentation(x, y, z)
    sph = coord.SphericalRepresentation.from_cartesian(cart)

    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)

    return r, colat, lon


@chk.check(
    dict(
        x=chk.accept_any(chk.is_real, chk.has_reals),
        y=chk.accept_any(chk.is_real, chk.has_reals),
        z=chk.accept_any(chk.is_real, chk.has_reals),
    )
)
def cart2eq(x, y, z):
    """
    Cartesian coordinates to Equatorial coordinates.

    Parameters
    ----------
    x : float or :py:class:`~numpy.ndarray`
        X coordinate.
    y : float or :py:class:`~numpy.ndarray`
        Y coordinate.
    z : float or :py:class:`~numpy.ndarray`
        Z coordinate.

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius.

    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].

    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.math.sphere.transform import cart2eq

    .. doctest::

       >>> r, lat, lon = cart2eq(0, 1 / np.sqrt(2), 1 / np.sqrt(2))

       >>> np.around(r, 2)
       1.0

       >>> np.around(np.rad2deg(lat), 2)
       45.0

       >>> np.around(np.rad2deg(lon), 2)
       90.0
    """
    r, colat, lon = cart2pol(x, y, z)
    lat = (np.pi / 2) - colat
    return r, lat, lon
