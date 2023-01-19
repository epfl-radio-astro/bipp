# ##############################################################################
# grid.py
# =======
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# ##############################################################################

"""
Tesselation schemes.
"""

import astropy.coordinates as coord
import astropy.units as u
import healpy
import numpy as np
import scipy.linalg as linalg

import bipp.imot_tools.math.linalg as ilinalg
import bipp.imot_tools.math.sphere.transform as transform
import bipp.imot_tools.util.argcheck as chk


@chk.check(
    dict(
        direction=chk.require_all(chk.has_reals, chk.has_shape([3])),
        FoV=chk.is_real,
        size=chk.require_all(chk.has_integers, chk.has_shape([2])),
    )
)
def spherical(direction, FoV, size):
    """
    Spherical pixel grid.

    Parameters
    ----------
    direction : :py:class:`~numpy.ndarray`
        (3,) vector around which the grid is centered.
    FoV : float
        Span of the grid [rad] centered at `direction`.
    size : :py:class:`~numpy.ndarray`
        (N_height, N_width)

        The grid will consist of `N_height` concentric circles around `direction`, each containing
        `N_width` pixels.

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_height, N_width) pixel grid.
    """
    direction = np.array(direction, dtype=float)
    direction /= linalg.norm(direction)

    if not (0 < np.rad2deg(FoV) <= 179):
        raise ValueError("Parameter[FoV] must be in (0, 179] degrees.")

    size = np.array(size, copy=False)
    if np.any(size <= 0):
        raise ValueError("Parameter[size] must contain positive entries.")

    N_height, N_width = size
    colat, lon = np.meshgrid(
        np.linspace(0, FoV / 2, N_height),
        np.linspace(0, 2 * np.pi, N_width),
        indexing="ij",
    )
    XYZ = transform.pol2cart(1, colat, lon)

    # Center grid at 'direction'
    _, dir_colat, _ = transform.cart2pol(*direction)
    R_axis = np.cross([0, 0, 1], direction)
    if np.allclose(R_axis, 0):
        # R_axis is in span(E_z), so we must manually set R
        R = np.eye(3)
        if direction[2] < 0:
            R[2, 2] = -1
    else:
        R = ilinalg.rot(axis=R_axis, angle=dir_colat)

    XYZ = np.tensordot(R, XYZ, axes=1)
    return XYZ


@chk.check(
    dict(
        direction=chk.require_all(chk.has_reals, chk.has_shape([3])),
        FoV=chk.is_real,
        size=chk.require_all(chk.has_integers, chk.has_shape([2])),
    )
)
def uniform(direction, FoV, size):
    """
    Uniform pixel grid.

    Parameters
    ----------
    direction : :py:class:`~numpy.ndarray`
        (3,) vector around which the grid is centered.
    FoV : float
        Span of the grid [rad] centered at `direction`.
    size : array-like(int)
        (N_height, N_width)

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_height, N_width) pixel grid.
    """
    direction = np.array(direction, dtype=float)
    direction /= linalg.norm(direction)

    if not (0 < np.rad2deg(FoV) <= 179):
        raise ValueError("Parameter[FoV] must be in (0, 179] degrees.")

    size = np.array(size, copy=False)
    if np.any(size <= 0):
        raise ValueError("Parameter[size] must contain positive entries.")

    N_height, N_width = size
    lim = np.sin(FoV / 2)
    Y, X = np.meshgrid(
        np.linspace(-lim, lim, N_height), np.linspace(-lim, lim, N_width), indexing="ij"
    )
    Z = 1 - X**2 - Y**2
    X[Z < 0], Y[Z < 0], Z[Z < 0] = 0, 0, 0
    Z = np.sqrt(Z)
    XYZ = np.stack([X, Y, Z], axis=0)

    # Center grid at 'direction'
    _, dir_colat, dir_lon = transform.cart2pol(*direction)
    R1 = ilinalg.rot(axis=[0, 0, 1], angle=dir_lon)
    R2_axis = np.cross([0, 0, 1], direction)
    if np.allclose(R2_axis, 0):
        # R2_axis is in span(E_z), so we must manually set R2.
        R2 = np.eye(3)
        if direction[2] < 0:
            R2[2, 2] = -1
    else:
        R2 = ilinalg.rot(axis=R2_axis, angle=dir_colat)
    R = R2 @ R1

    XYZ = np.tensordot(R, XYZ, axes=1)
    return XYZ


@chk.check(
    dict(
        N=chk.is_integer,
        direction=chk.allow_None(chk.require_all(chk.has_reals, chk.has_shape([3]))),
        FoV=chk.allow_None(chk.is_real),
    )
)
def equal_angle(N, direction=None, FoV=None):
    r"""
    (Region-limited) open grid of Equal-Angle sample-points on the sphere.

    Parameters
    ----------
    N : int
        Order of the grid, i.e. there will be :math:`4 (N + 1)^{2}` points on the sphere.
    direction : :py:class:`~numpy.ndarray`
        (3,) vector around which the grid is centered.
        If :py:obj:`None`, then the grid covers the entire sphere.
    FoV : float
        Span of the grid [rad] centered at `direction`.
        This parameter is ignored if `direction` left unspecified.

    Returns
    -------
    q : :py:class:`~numpy.ndarray`
        (N_height,) polar indices.

    l : :py:class:`~numpy.ndarray`
        (N_width,) azimuthal indices.

    colat : :py:class:`~numpy.ndarray`
        (N_height, 1) polar angles [rad].
        `N_height == 2N+2` in the whole-sphere case.

    lon : :py:class:`~numpy.ndarray`
        (1, N_width) azimuthal angles [rad].
        `N_width == 2N+2` in the whole-sphere case.

    Examples
    --------
    Sampling a zonal function :math:`f(r): \mathbb{S}^{2} \to \mathbb{C}` of order :math:`N` on the
    sphere:

    .. testsetup::

       import numpy as np

       from imot_tools.math.sphere.grid import equal_angle

    .. doctest::

       >>> N = 3
       >>> _, _, colat, lon = equal_angle(N)

       >>> np.around(colat, 2)
       array([[0.2 ],
              [0.59],
              [0.98],
              [1.37],
              [1.77],
              [2.16],
              [2.55],
              [2.95]])
       >>> np.around(lon, 2)
       array([[0.  , 0.79, 1.57, 2.36, 3.14, 3.93, 4.71, 5.5 ]])

    Sampling a zonal function :math:`f(r): \mathbb{S}^{2} \to \mathbb{C}` of order :math:`N` on
    *part* of the sphere:

    .. doctest::

       >>> N = 3
       >>> direction = np.r_[0, 1, 0]
       >>> FoV = np.deg2rad(90)
       >>> q, l, colat, lon = equal_angle(N, direction, FoV)

       >>> q
       array([2, 3, 4, 5])

       >>> np.around(colat, 2)
       array([[0.98],
              [1.37],
              [1.77],
              [2.16]])

       >>> l
       array([1, 2, 3])

       >>> np.around(lon, 2)
       array([[0.79, 1.57, 2.36]])

    Notes
    -----
    * The sample positions on the unit sphere are given (in radians) by [1]_:

    .. math::

       \theta_{q} & = \frac{\pi}{2 N + 2} \left( q + \frac{1}{2} \right), \qquad & q \in \{ 0, \ldots, 2 N + 1 \},

       \phi_{l} & = \frac{2 \pi}{2N + 2} l, \qquad & l \in \{ 0, \ldots, 2 N + 1 \}.

    * Longitudinal range may be erroneous if direction too close to [1, 0, 0].

    .. [1] B. Rafaely, "Fundamentals of Spherical Array Processing", Springer 2015
    """
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)

        if np.allclose(np.cross([0, 0, 1], direction), 0):
            raise ValueError(
                "Generating Equal-Angle grids centered at poles currently not supported."
            )
            # Why? Because the grid layout is spatially incorrect in this degenerate case.

        if FoV is not None:
            if not (0 < np.rad2deg(FoV) <= 179):
                raise ValueError("Parameter[FoV] must be in (0, 179] degrees.")
        else:
            raise ValueError(
                "Parameter[FoV] must be specified if Parameter[direction] provided."
            )

    if N <= 0:
        raise ValueError("Parameter[N] must be non-negative.")

    def ea_sample(N: int):
        _2N2 = 2 * N + 2
        q, l = np.ogrid[:_2N2, :_2N2]

        colat = (np.pi / _2N2) * (0.5 + q)
        lon = (2 * np.pi / _2N2) * l
        return colat, lon

    colat_full, lon_full = ea_sample(N)
    q_full = np.arange(colat_full.size)
    l_full = np.arange(lon_full.size)

    if direction is None:  # full-sphere case
        return q_full, l_full, colat_full, lon_full
    else:
        _, dir_colat, dir_lon = transform.cart2pol(*direction)
        lim_lon = dir_lon + (FoV / 2) * np.r_[-1, 1]
        lim_lon = coord.Angle(lim_lon * u.rad).wrap_at(360 * u.deg).to_value(u.rad)
        lim_colat = dir_colat + (FoV / 2) * np.r_[-1, 1]
        lim_colat = (
            max(np.deg2rad(0.5), lim_colat[0]),
            min(lim_colat[1], np.deg2rad(179.5)),
        )

        q_mask = (lim_colat[0] <= colat_full) & (colat_full <= lim_colat[1])
        if lim_lon[0] < lim_lon[1]:
            l_mask = (lim_lon[0] <= lon_full) & (lon_full <= lim_lon[1])
        else:
            l_mask = (lim_lon[0] <= lon_full) | (lon_full <= lim_lon[1])
        q_mask = np.reshape(q_mask, (-1,))
        l_mask = np.reshape(l_mask, (-1,))

        q, l = q_full[q_mask], l_full[l_mask]
        colat, lon = colat_full[q_mask, :], lon_full[:, l_mask]
        return q, l, colat, lon


@chk.check(
    dict(
        N=chk.is_integer,
        direction=chk.allow_None(chk.require_all(chk.has_reals, chk.has_shape([3]))),
        FoV=chk.allow_None(chk.is_real),
    )
)
def fibonacci(N, direction=None, FoV=None):
    r"""
    (Region-limited) near-uniform sampling on the sphere.

    Parameters
    ----------
    N : int
        Order of the grid, i.e. there will be :math:`4 (N + 1)^{2}` points on the sphere.
    direction : :py:class:`~numpy.ndarray`
        (3,) vector around which the grid is centered.
        If :py:obj:`None`, then the grid covers the entire sphere.
    FoV : float
        Span of the grid [rad] centered at `direction`.
        This parameter is ignored if `direction` left unspecified.

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_px) sample points.
        `N_px == 4*(N+1)**2` if `direction` left unspecified.

    Examples
    --------
    Sampling a zonal function :math:`f(r): \mathbb{S}^{2} \to \mathbb{C}` of order :math:`N` on the
    sphere:

    .. testsetup::

       import numpy as np

       from imot_tools.math.sphere.grid import fibonacci

    .. doctest::

       >>> N = 2
       >>> XYZ = fibonacci(N)

       >>> np.around(XYZ, 2)
       array([[ 0.23, -0.29,  0.04,  0.36, -0.65,  0.61, -0.2 , -0.37,  0.8 ,
               -0.81,  0.39,  0.28, -0.82,  0.95, -0.56, -0.13,  0.76, -1.  ,
                0.71, -0.05, -0.63,  0.97, -0.79,  0.21,  0.46, -0.87,  0.8 ,
               -0.33, -0.27,  0.68, -0.7 ,  0.36,  0.1 , -0.4 ,  0.4 , -0.16],
              [ 0.  , -0.27,  0.51, -0.47,  0.12,  0.39, -0.74,  0.72, -0.29,
               -0.34,  0.82, -0.89,  0.48,  0.21, -0.8 ,  0.98, -0.64, -0.04,
                0.71, -1.  ,  0.76, -0.13, -0.55,  0.93, -0.81,  0.28,  0.37,
               -0.78,  0.76, -0.36, -0.18,  0.56, -0.58,  0.31,  0.03, -0.17],
              [ 0.97,  0.92,  0.86,  0.81,  0.75,  0.69,  0.64,  0.58,  0.53,
                0.47,  0.42,  0.36,  0.31,  0.25,  0.19,  0.14,  0.08,  0.03,
               -0.03, -0.08, -0.14, -0.19, -0.25, -0.31, -0.36, -0.42, -0.47,
               -0.53, -0.58, -0.64, -0.69, -0.75, -0.81, -0.86, -0.92, -0.97]])

    Sampling a zonal function :math:`f(r): \mathbb{S}^{2} \to \mathbb{C}` of order :math:`N` on
    *part* of the sphere:

    .. doctest::

       >>> N = 2
       >>> direction = np.r_[1, 0, 0]
       >>> FoV = np.deg2rad(90)
       >>> XYZ = fibonacci(N, direction, FoV)

       >>> np.around(XYZ, 2)
       array([[ 0.8 ,  0.95,  0.76,  0.71,  0.97,  0.8 ],
              [-0.29,  0.21, -0.64,  0.71, -0.13,  0.37],
              [ 0.53,  0.25,  0.08, -0.03, -0.19, -0.47]])

    Notes
    -----
    The sample positions on the unit sphere are given (in radians) by [2]_:

    .. math::

       \cos(\theta_{q}) & = 1 - \frac{2 q + 1}{4 (N + 1)^{2}}, \qquad & q \in \{ 0, \ldots, 4 (N + 1)^{2} - 1 \},

       \phi_{q} & = \frac{4 \pi}{1 + \sqrt{5}} q, \qquad & q \in \{ 0, \ldots, 4 (N + 1)^{2} - 1 \}.


    .. [2] B. Rafaely, "Fundamentals of Spherical Array Processing", Springer 2015
    """
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)

        if FoV is not None:
            if not (0 < np.rad2deg(FoV) < 360):
                raise ValueError("Parameter[FoV] must be in (0, 360) degrees.")
        else:
            raise ValueError(
                "Parameter[FoV] must be specified if Parameter[direction] provided."
            )

    if N < 0:
        raise ValueError("Parameter[N] must be non-negative.")

    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)

    colat = np.arccos(1 - (2 * n + 1) / N_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    XYZ = np.stack(transform.pol2cart(1, colat, lon), axis=0)

    if direction is not None:  # region-limited case.
        # TODO: highly inefficient to generate the grid this way!
        min_similarity = np.cos(FoV / 2)
        mask = (direction @ XYZ) >= min_similarity
        XYZ = XYZ[:, mask]

    return XYZ


@chk.check(
    dict(
        N=chk.is_integer,
        direction=chk.allow_None(chk.require_all(chk.has_reals, chk.has_shape([3]))),
        FoV=chk.allow_None(chk.is_real),
    )
)
def healpix(N, direction=None, FoV=None):
    r"""
    (Region-limited) HEALPix sampling on the sphere.

    Parameters
    ----------
    N : int
        Order of the grid, i.e. there will be :math:`3 (N + 1)^{2}` points on the sphere.
    direction : :py:class:`~numpy.ndarray`
        (3,) vector around which the grid is centered.
        If :py:obj:`None`, then the grid covers the entire sphere.
    FoV : float
        Span of the grid [rad] centered at `direction`.
        This parameter is ignored if `direction` left unspecified.

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_px) sample points.
        `N_px == 3*(N+1)**2` if `direction` left unspecified. In this case pixels are RING-ordered.

    Examples
    --------
    Sampling a zonal function :math:`f(r): \mathbb{S}^{2} \to \mathbb{C}` of order :math:`N` on
    *part* of the sphere:

    .. testsetup::

       import numpy as np

       from imot_tools.math.sphere.grid import healpix

    .. doctest::

       >>> N = 10
       >>> direction = np.r_[1, 0, 0]
       >>> FoV = np.deg2rad(90)
       >>> XYZ = healpix(N, direction, FoV)

       >>> np.around(XYZ, 2)
       array([[ 0.74,  0.74,  0.85,  0.8 ,  0.8 ,  0.91,  0.82,  0.82,  0.91,
                0.96,  0.92,  0.78,  0.78,  0.92,  0.98,  0.88,  0.88,  0.98,
                1.  ,  0.95,  0.81,  0.81,  0.95,  0.98,  0.88,  0.88,  0.98,
                0.96,  0.92,  0.78,  0.78,  0.92,  0.91,  0.82,  0.82,  0.91,
                0.85,  0.8 ,  0.8 ,  0.74,  0.74],
              [ 0.12, -0.12,  0.  ,  0.26, -0.26,  0.14,  0.42, -0.42, -0.14,
                0.  ,  0.3 ,  0.57, -0.57, -0.3 ,  0.16,  0.45, -0.45, -0.16,
                0.  ,  0.31,  0.59, -0.59, -0.31,  0.16,  0.45, -0.45, -0.16,
                0.  ,  0.3 ,  0.57, -0.57, -0.3 ,  0.14,  0.42, -0.42, -0.14,
                0.  ,  0.26, -0.26,  0.12, -0.12],
              [ 0.67,  0.67,  0.53,  0.53,  0.53,  0.4 ,  0.4 ,  0.4 ,  0.4 ,
                0.27,  0.27,  0.27,  0.27,  0.27,  0.13,  0.13,  0.13,  0.13,
                0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.13, -0.13, -0.13, -0.13,
               -0.27, -0.27, -0.27, -0.27, -0.27, -0.4 , -0.4 , -0.4 , -0.4 ,
               -0.53, -0.53, -0.53, -0.67, -0.67]])

    Notes
    -----
    The `HEALPix <https://healpix.jpl.nasa.gov/>`_ scheme is defined in [3]_.

    .. [3] Gorski, "HEALPix - A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere"
    """
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)

        if FoV is not None:
            if not (0 < np.rad2deg(FoV) < 360):
                raise ValueError("Parameter[FoV] must be in (0, 360) degrees.")
        else:
            raise ValueError(
                "Parameter[FoV] must be specified if Parameter[direction] provided."
            )

    if N < 0:
        raise ValueError("Parameter[N] must be non-negative.")

    N_side = (N + 1) // 2
    N_px = 12 * (N_side**2)
    n = np.arange(N_px)

    XYZ = np.stack(healpy.pix2vec(N_side, n), axis=0)

    if direction is not None:  # region-limited case.
        # TODO: highly inefficient to generate the grid this way!
        min_similarity = np.cos(FoV / 2)
        mask = (direction @ XYZ) >= min_similarity
        XYZ = XYZ[:, mask]

    return XYZ
