# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Phased-Array Signal Processing tools.
"""

import numpy as np
import scipy.linalg as linalg

import bipp.imot_tools.util.argcheck as chk
import bipp.imot_tools.math.special as special


@chk.check(dict(XYZ=chk.has_reals, R=chk.has_reals, wl=chk.is_real))
def steering_operator(XYZ, R, wl):
    r"""
    Steering matrix.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian array geometry.
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points in :math:`\mathbb{S}^{2}`.
    wl : float
        Wavelength [m].

    Returns
    -------
    A : :py:class:`~numpy.ndarray`
        (N_antenna, N_px) steering matrix.

    Notes
    -----
    The steering matrix is defined as:

    .. math:: {\bf{A}} = \exp \left( -j \frac{2 \pi}{\lambda} {\bf{P}}^{T} {\bf{R}} \right),

    where :math:`{\bf{P}} \in \mathbb{R}^{3 \times N_{\text{antenna}}}` and
    :math:`{\bf{R}} \in \mathbb{R}^{3 \times N_{\text{px}}}`.
    """
    if wl <= 0:
        raise ValueError("Parameter[wl] must be positive.")

    scale = 2 * np.pi / wl
    A = np.exp((-1j * scale * XYZ.T) @ R)
    return A


@chk.check(dict(XYZ=chk.has_reals, wl=chk.is_real))
def nyquist_rate(XYZ, wl):
    """
    Order of imageable complex plane-waves by an instrument.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian array geometry.
    wl : float
        Wavelength [m]

    Returns
    -------
    N : int
        Maximum order of complex plane waves that can be imaged by the instrument.
    """
    baseline = linalg.norm(XYZ[:, np.newaxis, :] - XYZ[:, :, np.newaxis], axis=0)

    N = special.spherical_jn_series_threshold((2 * np.pi / wl) * baseline.max())
    return N
