# #############################################################################
# func.py
# =======
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
1D functions not available in `SciPy <https://www.scipy.org/>`_.
"""

import warnings

import numpy as np
import scipy.interpolate as interpolate
import scipy.linalg as linalg
import scipy.special as special

import bipp.imot_tools.util.argcheck as chk
import bipp.imot_tools.math.special as ispecial
import bipp.numpy_compat as npc


class Tukey:
    r"""
    Parameterized Tukey function.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.math.func import Tukey

    .. doctest::

       >>> tukey = Tukey(T=1, beta=0.5, alpha=0.25)

       >>> sample_points = np.linspace(0, 1, 25).reshape(5, 5)  # any shape
       >>> amplitudes = tukey(sample_points)
       >>> np.around(amplitudes, 2)
       array([[0.  , 0.25, 0.75, 1.  , 1.  ],
              [1.  , 1.  , 1.  , 1.  , 1.  ],
              [1.  , 1.  , 1.  , 1.  , 1.  ],
              [1.  , 1.  , 1.  , 1.  , 1.  ],
              [1.  , 1.  , 0.75, 0.25, 0.  ]])

    Notes
    -----
    The Tukey function is defined as:

    .. math::

       \text{Tukey}(T, \beta, \alpha)(\varphi): \mathbb{R} & \to [0, 1] \\
       \varphi & \to
       \begin{cases}
           % LINE 1
           \sin^{2} \left( \frac{\pi}{T \alpha}
                    \left[ \frac{T}{2} - \beta + \varphi \right] \right) &
           0 \le \frac{T}{2} - \beta + \varphi < \frac{T \alpha}{2} \\
           % LINE 2
           1 &
           \frac{T \alpha}{2} \le \frac{T}{2} - \beta +
           \varphi \le T - \frac{T \alpha}{2} \\
           % LINE 3
           \sin^{2} \left( \frac{\pi}{T \alpha}
                    \left[ \frac{T}{2} + \beta - \varphi \right] \right) &
           T - \frac{T \alpha}{2} < \frac{T}{2} - \beta + \varphi \le T \\
           % LINE 4
           0 &
           \text{otherwise.}
       \end{cases}
    """

    @chk.check(dict(T=chk.is_real, beta=chk.is_real, alpha=chk.is_real))
    def __init__(self, T, beta, alpha):
        """
        Parameters
        ----------
        T : float
            Function support.
        beta : float
            Function mid-point.
        alpha : float
           Normalized decay-rate.
        """
        self._beta = beta

        if not (T > 0):
            raise ValueError("Parameter[T] must be positive.")
        self._T = T

        if not (0 <= alpha <= 1):
            raise ValueError("Parameter[alpha] must be in [0, 1].")
        self._alpha = alpha

    @chk.check("x", chk.accept_any(chk.is_real, chk.has_reals))
    def __call__(self, x):
        """
        Sample the Tukey(T, beta, alpha) function.

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            Sample points.

        Returns
        -------
        Tukey(T, beta, alpha)(x) : :py:class:`~numpy.ndarray`
        """
        x = npc.asarray(x)

        y = x - self._beta + self._T / 2
        left_lim = float(self._T * self._alpha / 2)
        right_lim = float(self._T - (self._T * self._alpha / 2))

        ramp_up = (0 <= y) & (y < left_lim)
        body = (left_lim <= y) & (y <= right_lim)
        ramp_down = (right_lim < y) & (y <= self._T)

        amplitude = np.zeros_like(x)
        amplitude[body] = 1
        if not np.isclose(self._alpha, 0):
            amplitude[ramp_up] = (
                np.sin(np.pi / (self._T * self._alpha) * y[ramp_up]) ** 2
            )
            amplitude[ramp_down] = (
                np.sin(np.pi / (self._T * self._alpha) * (self._T - y[ramp_down])) ** 2
            )
        return amplitude


class SphericalDirichlet:
    r"""
    Parameterized spherical Dirichlet kernel.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.math.func import SphericalDirichlet

    .. doctest::

       >>> N = 4
       >>> f = SphericalDirichlet(N)

       >>> sample_points = np.linspace(-1, 1, 25).reshape(5, 5)  # any shape
       >>> amplitudes = f(sample_points)
       >>> np.around(amplitudes, 2)
       array([[ 0.4 ,  0.08, -0.1 , -0.17, -0.17],
              [-0.13, -0.05,  0.03,  0.1 ,  0.16],
              [ 0.19,  0.18,  0.15,  0.09,  0.  ],
              [-0.1 , -0.19, -0.27, -0.3 , -0.27],
              [-0.15,  0.11,  0.52,  1.13,  1.99]])

    When only interested in kernel values close to 1, the approximation method provides significant
    speedups, at the cost of approximation error in values far from 1:

    .. doctest::

       N = 11
       f_exact = SphericalDirichlet(N)
       f_approx = SphericalDirichlet(N, approx=True)

       x = np.linspace(-1, 1, 2000)
       e_y = f_exact(x)
       a_y = f_approx(x)
       rel_err = np.abs((e_y - a_y) / e_y)

       fig, ax = plt.subplots(nrows=2)
       ax[0].plot(x, e_y, 'r')
       ax[0].plot(x, a_y, 'b')
       ax[0].legend(['exact', 'approx'])
       ax[0].set_title('Dirichlet Kernel')

       ax[1].plot(x, rel_err)
       ax[1].set_title('Relative Error (Exact vs. Approx)')

       fig.show()

    .. image:: _img/sph_dirichlet_example.png

    Notes
    -----
    The spherical Dirichlet function :math:`K_{N}(t): [-1, 1] \to \mathbb{R}` is defined as:

    .. math:: K_{N}(t) = \frac{N+1}{4\pi} \frac{P_{N+1}(t) - P_{N}(t)}{t - 1},

    where :math:`P_{N}(t)` is the `Legendre polynomial <https://en.wikipedia.org/wiki/Legendre_polynomials>`_
    of order :math:`N`.
    """

    @chk.check(dict(N=chk.is_integer, approx=chk.is_boolean))
    def __init__(self, N, approx=False):
        """
        Parameters
        ----------
        N : int
            Kernel order.
        approx : bool
            Approximate kernel using cubic-splines.

            This method provides extremely reliable estimates of :math:`K_{N}(t)` in the vicinity of
            1 where the function's main sidelobes are found. Values outside the vicinity smoothly
            converge to 0.

            Only works for `N` greater than 10.
        """
        if N < 0:
            raise ValueError("Parameter[N] must be non-negative.")
        self._N = N

        if (approx is True) and (N <= 10):
            raise ValueError("Cannot use approximation method if Parameter[N] <= 10.")
        self._approx = approx

        if approx is True:  # Fit cubic-spline interpolator.
            N_samples = 10**3

            # Find interval LHS after which samples will be evaluated exactly.
            theta_max = np.pi
            while True:
                x = np.linspace(0, theta_max, N_samples)
                cx = np.cos(x)
                cy = self._exact_kernel(cx)
                zero_cross = np.diff(np.sign(cy))
                N_cross = np.abs(np.sign(zero_cross)).sum()

                if N_cross > 10:
                    theta_max /= 2
                else:
                    break

            window = Tukey(T=2 - 2 * np.cos(2 * theta_max), beta=1, alpha=0.5)

            x = np.r_[
                np.linspace(
                    np.cos(theta_max * 2), np.cos(theta_max), N_samples, endpoint=False
                ),
                np.linspace(np.cos(theta_max), 1, N_samples),
            ]
            y = self._exact_kernel(x) * window(x)
            self.__cs_interp = interpolate.interp1d(
                x, y, kind="cubic", bounds_error=False, fill_value=0
            )

    @chk.check("x", chk.accept_any(chk.is_real, chk.has_reals))
    def __call__(self, x):
        r"""
        Sample the order-N spherical Dirichlet kernel.

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            Values at which to compute :math:`K_{N}(x)`.

        Returns
        -------
        K_N(x) : :py:class:`~numpy.ndarray`
        """
        if chk.is_scalar(x):
            x = np.array([x], dtype=float)
        else:
            x = npc.asarray(x, dtype=float)

        if not np.all((-1 <= x) & (x <= 1)):
            raise ValueError("Parameter[x] must lie in [-1, 1].")

        if self._approx is True:
            f = self._approx_kernel
        else:
            f = self._exact_kernel

        amplitude = f(x)
        return amplitude

    # @chk.check('x', chk.accept_any(chk.is_real, chk.has_reals))
    def _exact_kernel(self, x):
        amplitude = special.eval_legendre(self._N + 1, x) - special.eval_legendre(
            self._N, x
        )
        with warnings.catch_warnings():
            # The kernel is so condensed near 1 at high N that np.isclose()
            # does a terrible job at letting us manually treat values close to
            # the upper limit.
            # The best way to implement K_N(t) is to let the floating point
            # division fail and then replace NaNs.
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            amplitude /= x - 1
        amplitude[np.isnan(amplitude)] = self._N + 1

        amplitude *= (self._N + 1) / (4 * np.pi)
        return amplitude

    # @chk.check('x', chk.accept_any(chk.is_real, chk.has_reals))
    def _approx_kernel(self, x):
        amplitude = self.__cs_interp(x)
        return amplitude


class Kent:
    r"""
    Parameterized `Kent <https://en.wikipedia.org/wiki/Kent_distribution>`_ distribution, also known
    as :math:`\text{FB}_{5}`, the 5-parameter Fisher-Bingham distribution.

    The density of :math:`\text{FB}_{5}(k, \beta, \gamma_{1}, \gamma_{2}, \gamma_{3})` is given by

    .. math::

       f(x) & = \frac{1}{c(k,\beta)} \exp\left( \gamma_{1}^{\intercal} x + \frac{\beta}{2} \left[ (\gamma_{2}^{\intercal} x)^{2} - (\gamma_{3}^{\intercal} x)^{2} \right] - 1 \right)^{k},

       c(k, \beta) & = \sqrt{\frac{8 \pi}{k}} \sum_{j \ge 0} B\left(j + \frac{1}{2}, \frac{1}{2}\right) \beta^{2 j} I_{2 j + \frac{1}{2}}^{e}(k),

    where :math:`\beta \in [0, 1)` determines the distribution's ellipticity, :math:`B(\cdot, \cdot)`
    denotes the Beta function, and :math:`I_{v}^{e}(z) = I_{v}(z) e^{-|\Re{\{z\}}|}` is the
    exponentially-scaled modified Bessel function of the first kind.
    """

    @chk.check(
        dict(
            k=chk.is_real,
            beta=chk.is_real,
            g1=chk.require_all(chk.has_reals, chk.has_shape([3])),
            a=chk.require_all(chk.has_reals, chk.has_shape([3])),
        )
    )
    def __init__(self, k, beta, g1, a):
        r"""
        Parameters
        ----------
        k : float
            Scale parameter.
        beta : float
            Ellipticity in [0, 1[.
        g1 : :py:class:`~numpy.ndarray`
            (3,) mean direction vector :math:`\gamma_{1}`.
        a : :py:class:`~numpy.ndarray`
            (3,) direction of major axis.

            This is *not* the same thing as :math:`\gamma_{2}`!

        Notes
        -----
        :math:`\gamma_{1}` and `a` are sufficient statistics to determine the directional vectors
        :math:`\gamma_{2}, \gamma_{3} \in \mathbb{R}^{3}`.
        """
        if k <= 0:
            raise ValueError("Parameter[k] must be positive.")
        self._k = k

        if not (0 <= beta < 1):
            raise ValueError("Parameter[beta] must lie in [0, 1).")
        self._beta = beta

        self._g1 = npc.asarray(g1) / linalg.norm(g1)
        a = npc.asarray(a) / linalg.norm(a)
        if np.allclose(self._g1, a):
            raise ValueError("Parameters[g1, a] must not be colinear.")

        # Find major/minor axes (g2,g3)
        Q, _ = linalg.qr(np.stack([self._g1, a], axis=1))
        self._g2 = Q[:, 1]
        self._g3 = np.cross(self._g1, self._g2)

        # Buffered attributes
        ive_threshold = ispecial.ive_threshold(k)
        j = np.arange((ive_threshold - 0.5) // 2 + 2)
        self._cst = np.sqrt(8 * np.pi / k) * np.sum(
            special.beta(j + 0.5, 0.5) * (beta ** (2 * j)) * special.ive(2 * j + 0.5, k)
        )

    @chk.check("x", chk.has_reals)
    def __call__(self, x):
        """
        Density of the distribution at sample points.

        Parameters
        ----------
        x : :py:class:`numpy.ndarray`
            (N, 3) values at which to determine the pdf.

        Returns
        -------
        pdf : :py:class:`~numpy.ndarray`
            (N,) densities.
        """
        x = np.array(x, dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis]
        elif x.ndim == 2:
            pass
        else:
            raise ValueError("Parameter[x] must have shape (N, 3).")

        N = len(x)
        if not chk.has_shape([N, 3])(x):
            raise ValueError("Parameter[x] must have shape (N, 3).")
        x /= linalg.norm(x, axis=1, keepdims=True)

        pdf = (
            np.exp(
                (x @ self._g1)
                - 1
                + 0.5 * self._beta * ((x @ self._g2) ** 2 - (x @ self._g3) ** 2)
            )
            ** self._k
        )
        pdf /= self._cst
        return pdf

    @classmethod
    @chk.check(dict(k=chk.is_real, beta=chk.is_real, eps=chk.is_real))
    def angular_support(cls, k, beta, eps=1e-2):
        r"""
        Pdf angular span.

        For a given parameterization :math:`k, \beta, \gamma_{1}, \gamma_{2}, \gamma_{3}` of
        :math:`\text{FB}_{5}`, :py:meth:`~imot_tools.math.func.Kent.angular_support` returns
        the angular separation between :math:`\gamma_{1}` and a point :math:`r` along
        :math:`\gamma_{2}` on the sphere where :math:`\epsilon f(\gamma_{1}) = f(r)`.

        The solution is given by :math:`\theta^{\ast} = \arg\min_{\theta > 0} \cos\theta + \frac{\beta}{2}\sin^{2}\theta \le 1 + \frac{1}{k} \ln\epsilon`.

        Parameters
        ----------
        k : float
            Scale parameter.
        beta : float
            Ellipticity in [0, 1[.
        eps : float
            Constant :math:`\epsilon` in ]0, 1[.

        Returns
        -------
        theta : float
            Angular separation [rad] between :math:`r` and :math:`\gamma_{1}`.
        """
        if k <= 0:
            raise ValueError("Parameter[k] must be positive.")

        if not (0 <= beta < 1):
            raise ValueError("Parameter[beta] must lie in [0, 1).")

        if not (0 < eps < 1):
            raise ValueError("Parameter[eps] must lie in (0, 1).")

        theta = np.linspace(0, np.pi, 1e5)
        lhs = np.cos(theta) + 0.5 * beta * np.sin(theta) ** 2
        rhs = 1 + np.log(eps) / k
        mask = lhs <= rhs

        if np.any(mask):
            support = theta[mask][0]
        else:
            support = np.pi
        return support

    @classmethod
    @chk.check(dict(alpha=chk.is_real, beta=chk.is_real, eps=chk.is_real))
    def min_scale(cls, alpha, beta, eps=1e-2):
        r"""
        Minimum scale parameter for desired concentration.

        For a given parameterization :math:`k, \beta, \gamma_{1}, \gamma_{2}, \gamma_{3}` of
        :math:`\text{FB}_{5}`, :py:meth:`~imot_tools.math.func.Kent.min_scale` returns the
        minimum value of :math:`k` such that a spherical cap :math:`S` with opening half-angle
        :math:`\alpha` centered at :math:`\gamma_{1}` contains the distribution's isoline of
        amplitude :math:`\epsilon f(\gamma_{1})`.

        The solution is given by :math:`k^{\ast} = \log \epsilon / \left( \cos\alpha + \frac{\beta}{2}\sin^{2}\alpha - 1 \right)`.

        Parameters
        ----------
        alpha : float
            Angular span [rad] of the density between :math:`\gamma_{1}` and a point :math:`r`
            along :math:`\gamma_{2}` on the sphere where :math:`f(r) = \epsilon f(\gamma_{1})`.
        beta : float
            Ellipticity in [0, 1[.
        eps : float
            Constant :math:`\epsilon` in ]0, 1[.

        Returns
        -------
        k : int
            scale parameter.
        """
        if not (0 < alpha <= np.pi):
            raise ValueError("Parameter[alpha] is out of bounds.")

        if not (0 <= beta < 1):
            raise ValueError("Parameter[beta] must lie in [0, 1).")

        if not (0 < eps < 1):
            raise ValueError("Parameter[eps] must lie in (0, 1).")

        denom = np.cos(alpha) + 0.5 * beta * np.sin(alpha) ** 2 - 1

        if np.isclose(denom, 0):
            k = np.inf
        else:
            k = np.abs(np.log(eps) / denom)
        return k
