# ##############################################################################
# interpolate.py
# ==============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# ##############################################################################

"""
Interpolation algorithms.
"""

import numpy as np
import scipy.sparse as sp
import tqdm

import bipp.imot_tools.math.func as func
import bipp.imot_tools.math.sphere.grid as grid
import bipp.imot_tools.math.sphere.transform as transform
import bipp.imot_tools.util.argcheck as chk


class Interpolator:
    r"""
    Interpolate order-limited zonal function from spatial samples.

    Computes :math:`f(r) = \sum_{q} \alpha_{q} f(r_{q}) K_{N}(\langle r, r_{q} \rangle)`, where
    :math:`r_{q} \in \mathbb{S}^{2}` are points from a spatial sampling scheme, :math:`K_{N}(\cdot)`
    is the spherical Dirichlet kernel of order :math:`N`, and the :math:`\alpha_{q}` are scaling
    factors tailored to the sampling scheme.
    """

    @chk.check(dict(N=chk.is_integer, approximate_kernel=chk.is_boolean))
    def __init__(self, N, approximate_kernel=False):
        r"""
        Parameters
        ----------
        N : int
            Order of the reconstructed zonal function.
        approximate_kernel : bool
            If :py:obj:`True`, pass the `approx` option to :py:class:`~imot_tools.math.func.SphericalDirichlet`.
        """
        if not (N > 0):
            raise ValueError("Parameter[N] must be positive.")
        self._N = N
        self._kernel_func = func.SphericalDirichlet(N, approximate_kernel)

    @chk.check(
        dict(
            weight=chk.has_reals,
            support=chk.has_reals,
            f=chk.accept_any(chk.has_reals, chk.has_complex),
            r=chk.has_reals,
            sparsity_mask=chk.allow_None(
                chk.require_all(
                    chk.is_instance(sp.spmatrix),
                    lambda _: np.issubdtype(_.dtype, np.bool_),
                )
            ),
        )
    )
    def __call__(self, weight, support, f, r, sparsity_mask=None):
        """
        Interpolate function samples at order `N`.

        Parameters
        ----------
        weight : :py:class:`~numpy.ndarray`
            (N_s,) weights to apply per support point.
        support : :py:class:`~numpy.ndarray`
            (3, N_s) critical support points.
        f : :py:class:`~numpy.ndarray`
            (L, N_s) zonal function values at support points. (float or complex)
        r : :py:class:`~numpy.ndarray`
            (3, N_px) evaluation points.
        sparsity_mask : :py:class:`~scipy.sparse.spmatrix`
            (N_s, N_px) sparsity mask (bool) to perform localized kernel evaluation.

        Returns
        -------
        f_interp : :py:class:`~numpy.ndarray`
            (L, N_px) function values at specified coordinates.
        """
        if not (weight.shape == (weight.size,)):
            raise ValueError("Parameter[weight] must have shape (N_s,).")
        N_s = weight.size

        if not (support.shape == (3, N_s)):
            raise ValueError("Parameter[support] must have shape (3, N_s).")

        L = len(f)
        if not (f.shape == (L, N_s)):
            raise ValueError("Parameter[f] must have shape (L, N_s).")

        if not ((r.ndim == 2) and (r.shape[0] == 3)):
            raise ValueError("Parameter[r] must have shape (3, N_px).")
        N_px = r.shape[1]

        if sparsity_mask is not None:
            if not (sparsity_mask.shape == (N_s, N_px)):
                raise ValueError(
                    "Parameter[sparsity_mask] must have shape (N_s, N_px)."
                )

        if sparsity_mask is None:  # Dense evaluation
            kernel = self._kernel_func(support.T @ r)
            beta = f * weight
            f_interp = beta @ kernel
        else:  # Sparse evaluation
            # Evaluate kernel
            row = sparsity_mask.row
            col = sparsity_mask.col
            dist = np.sum(support[:, row] * r[:, col], axis=0)
            ker = self._kernel_func(np.clip(dist, 0, 1))
            kernel = sp.coo_matrix((ker, (row, col)), shape=sparsity_mask.shape)

            kernel_T = kernel.T.tocsr()
            beta = f * weight
            f_interp = (kernel_T @ beta.T).T
        return f_interp


class EqualAngleInterpolator(Interpolator):
    r"""
    Interpolate order-limited zonal function from Equal-Angle samples.

    Computes :math:`f(r) = \sum_{q, l} \alpha_{q} f(r_{q, l}) K_{N}(\langle r, r_{q, l} \rangle)`,
    where :math:`r_{q, l} \in \mathbb{S}^{2}` are points from an Equal-Angle sampling scheme,
    :math:`K_{N}(\cdot)` is the spherical Dirichlet kernel of order :math:`N`, and the
    :math:`\alpha_{q}` are scaling factors tailored to an Equal-Angle sampling scheme.

    Examples
    --------
    Let :math:`\gamma_{N}(r): \mathbb{S}^{2} \to \mathbb{R}` be the order-:math:`N` approximation of
    :math:`\gamma(r) = \delta(r - r_{0})`:

    .. math::

       \gamma_{N}(r) = \frac{N + 1}{4 \pi} \frac{P_{N + 1}(\langle r, r_{0} \rangle) - P_{N}(\langle r, r_{0} \rangle)}{\langle r, r_{0} \rangle -1}.

    As :math:`\gamma_{N}` is order-limited, it can be exactly reconstructed from it's samples on an
    order-:math:`N` Equal-Angle grid:

    .. testsetup::

       import numpy as np

       from imot_tools.math.func import SphericalDirichlet
       from imot_tools.math.sphere.grid import equal_angle
       from imot_tools.math.sphere.interpolate import EqualAngleInterpolator
       from imot_tools.math.sphere.transform import pol2cart

       def gammaN(r, r0, N):
           similarity = np.tensordot(r0, r, axes=1)
           d_func = SphericalDirichlet(N)
           return d_func(similarity)

    .. doctest::

       # \gammaN Parameters
       >>> N = 3
       >>> r0 = np.array([1, 0, 0])

       # Solution at Nyquist resolution
       >>> colat_idx, lon_idx, colat_nyquist, lon_nyquist = equal_angle(N)
       >>> N_colat, N_lon = colat_nyquist.size, lon_nyquist.size
       >>> R_nyquist = pol2cart(1, colat_nyquist, lon_nyquist)
       >>> g_nyquist = gammaN(R_nyquist, r0, N)

       # Solution at high resolution
       >>> _, _, colat_dense, lon_dense = equal_angle(2 * N)
       >>> R_dense = pol2cart(1, colat_dense, lon_dense).reshape(3, -1)
       >>> g_exact = gammaN(R_dense, r0, N)

       >>> ea_interp = EqualAngleInterpolator(N)
       >>> g_interp = ea_interp(colat_idx,
       ...                      lon_idx,
       ...                      f=g_nyquist.reshape(1, N_colat, N_lon),
       ...                      r=R_dense)

       >>> np.allclose(g_exact, g_interp)
       True
    """

    @chk.check(dict(N=chk.is_integer, approximate_kernel=chk.is_boolean))
    def __init__(self, N, approximate_kernel=False):
        r"""
        Parameters
        ----------
        N : int
            Order of the reconstructed zonal function.
        approximate_kernel : bool
            If :py:obj:`True`, pass the `approx` option to :py:class:`~imot_tools.math.func.SphericalDirichlet`.
        """
        super().__init__(N, approximate_kernel)

    @chk.check(
        dict(
            colat_idx=chk.has_integers,
            lon_idx=chk.has_integers,
            f=chk.accept_any(chk.has_reals, chk.has_complex),
            r=chk.has_reals,
            sparsity_mask=chk.allow_None(
                chk.require_all(
                    chk.is_instance(sp.spmatrix),
                    lambda _: np.issubdtype(_.dtype, np.bool_),
                )
            ),
        )
    )
    def __call__(self, colat_idx, lon_idx, f, r, sparsity_mask=None):
        """
        Interpolate function samples at order `N`.

        Parameters
        ----------
        colat_idx : :py:class:`~numpy.ndarray`
            (N_colat,) polar support indices from :py:func:`~imot_tools.math.sphere.grid.equal_angle`.
        lon_idx : :py:class:`~numpy.ndarray`
            (N_lon,) azimuthal support indices from :py:func:`~imot_tools.math.sphere.grid.equal_angle`.
        f : :py:class:`~numpy.ndarray`
            (L, N_colat, N_lon) zonal function values at support points. (float or complex)
        r : :py:class:`~numpy.ndarray`
            (3, N_px) evaluation points.
        sparsity_mask : :py:class:`~scipy.sparse.spmatrix`
            (N_s, N_px) sparsity mask (bool) to perform localized kernel evaluation.
            The 0-th dimension has size N_s = N_colat * N_lon.

        Returns
        -------
        f_interp : :py:class:`~numpy.ndarray`
            (L, N_px) function values at specified coordinates.
        """
        N_colat = colat_idx.size
        if not (colat_idx.shape == (N_colat,)):
            raise ValueError("Parameter[colat_idx] must have shape (N_colat,).")

        N_lon = lon_idx.size
        if not (lon_idx.shape == (N_lon,)):
            raise ValueError("Parameter[lon_idx] must have shape (N_lon,).")

        L = len(f)
        if not (f.shape == (L, N_colat, N_lon)):
            raise ValueError("Parameter[f] must have shape (L, N_colat, N_lon).")

        if not ((r.ndim == 2) and (r.shape[0] == 3)):
            raise ValueError("Parameter[r] must have shape (3, N_px).")

        # Apply weights directly onto `f` to avoid memory blow-up.
        _, _, colat, lon = grid.equal_angle(self._N)
        a = np.arange(self._N + 1)
        weight = (
            np.sum(np.sin((2 * a + 1) * colat[colat_idx]) / (2 * a + 1), axis=1)
            * np.sin(colat[colat_idx, 0])
            * ((2 * np.pi) / ((self._N + 1) ** 2))
        )  # (N_colat,)
        fw = f * weight.reshape((1, N_colat, 1))  # (L, N_colat, N_lon)

        f_interp = super().__call__(
            weight=np.broadcast_to([1], (N_colat * N_lon,)),
            support=transform.pol2cart(1, colat[colat_idx, :], lon[:, lon_idx]).reshape(
                (3, -1)
            ),
            f=fw.reshape((L, -1)),
            r=r,
            sparsity_mask=sparsity_mask,
        )
        return f_interp


@chk.check(
    dict(
        N=chk.is_integer,
        R1=chk.require_all(chk.has_ndim(2), chk.has_reals),
        R2=chk.require_all(chk.has_ndim(2), chk.has_reals),
    )
)
def sparsity_mask(N, R1, R2):
    r"""
    Generate sparsity mask for fast kernel evaluation in
    :py:class:`~imot_tools.math.sphere.Interpolator` instances.

    Parameters
    ----------
    N : int
        Interpolation order.
    R1 : :py:class:`~numpy.ndarray`
        (3, N_1) Cartesian coordinates.
    R2 : :py:class:`~numpy.ndarray`
        (3, N_2) Cartesian coordinates.

    Returns
    -------
    S : :py:class:`~scipy.sparse.coo_matrix` (bool)
        (N_1, N_2) boolean mask where kernel must be evaluated.

    Notes
    -----
    Dense equivalent:

        S = (R1.T @ R2) >= threshold(N)
    """
    N_1 = R1.shape[1]
    if R1.shape != (3, N_1):
        raise ValueError("Parameter[R1] must be a (3, N_1) array.")
    N_2 = R2.shape[1]
    if R2.shape != (3, N_2):
        raise ValueError("Parameter[R2] must be a (3, N_2) array.")

    if N_1 < N_2:
        S = sparsity_mask(N, R2, R1).T
        return S
    else:
        # Compute cut-off threshold
        f = func.SphericalDirichlet(N, approx=True)
        x = np.linspace(-1, 1, 10**6)
        threshold = x[np.cumsum((f(x) / f(1)) ** 2).nonzero()].min()

        S = sp.dok_matrix((N_1, N_2), dtype=bool)
        L = np.zeros((N_2,), dtype=float)  # Temporary buffer
        idx = np.zeros((N_2,), dtype=bool)  # Temporary buffer
        for i in tqdm.tqdm(np.arange(N_1)):
            np.dot(R1[:, i], R2, out=L)
            np.greater_equal(L, threshold, out=idx)
            S[i, idx] = True
        S = sp.coo_matrix(S)
        return S
