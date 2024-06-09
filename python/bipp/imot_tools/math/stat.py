# #############################################################################
# stat.py
# =======
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Statistical functions not available in `SciPy <https://www.scipy.org/>`_.
"""

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats

import bipp.imot_tools.util.argcheck as chk


class RandomSampler:
    """
    Random number generator.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import scipy.linalg as linalg

       from imot_tools.math.stat import Wishart

       np.random.seed(0)

       def hermitian_array(N: int) -> np.ndarray:
           '''
           Construct a (N, N) Hermitian matrix.
           '''
           D = np.arange(1, N + 1)
           Rmtx = np.random.randn(N,N) + 1j * np.random.randn(N, N)
           Q, _ = linalg.qr(Rmtx)

           A = (Q * D) @ Q.conj().T
           return A

    .. doctest::

       >>> A = hermitian_array(N=4)  # random (N, N) PSD array.
       >>> n = 50
       >>> W = Wishart(A, n)
       >>> samples = W(N_sample=200)

       # Law of large numbers: test convergence to mean.
       >>> eps = 0.05
       >>> linalg.norm(A * n - np.mean(samples, axis=0)) < eps * linalg.norm(A * n)
       True
    """

    def __init__(self):
        """"""
        pass

    @chk.check("N_sample", chk.is_integer)
    def __call__(self, N_sample=1):
        """
        Generate random samples.

        Parameters
        ----------
        N_sample : int
            Number of samples to generate.

        Returns
        -------
        x : :py:class:`~numpy.ndarray`
            (N_sample, ...) samples.
        """
        raise NotImplementedError


class Wishart(RandomSampler):
    """
    `Wishart <https://en.wikipedia.org/wiki/Wishart_distribution>`_ distribution.
    """

    @chk.check(dict(V=chk.accept_any(chk.has_reals, chk.has_complex), n=chk.is_integer))
    def __init__(self, V, n):
        """
        Parameters
        ----------
        V : :py:class:`~numpy.ndarray`
            (p, p) positive-semidefinite scale matrix.
        n : int
            degrees of freedom.
        """
        super().__init__()

        V = np.array(V)
        p = len(V)

        if not (chk.has_shape([p, p])(V) and np.allclose(V, V.conj().T)):
            raise ValueError("Parameter[V] must be hermitian symmetric.")
        if not (n > p):
            raise ValueError(f"Parameter[n] must be greater than {p}.")

        self._V = V
        self._p = p
        self._n = n

        Vq = linalg.sqrtm(V)
        _, R = linalg.qr(Vq)
        self._L = R.conj().T

    @chk.check("N_sample", chk.is_integer)
    def __call__(self, N_sample=1):
        """
        Generate random samples.

        Parameters
        ----------
        N_sample : int
            Number of samples to generate.

        Returns
        -------
        x : :py:class:`~numpy.ndarray`
            (N_sample, p, p) samples.

        Notes
        -----
        The Wishart estimate is obtained using the `Bartlett Decomposition`_.

        .. _Bartlett Decomposition: https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition
        """
        if N_sample < 1:
            raise ValueError("Parameter[N_sample] must be positive.")

        A = np.zeros((N_sample, self._p, self._p))

        diag_idx = np.diag_indices(self._p)
        df = self._n * np.ones((N_sample, 1)) - np.arange(self._p)
        A[:, diag_idx[0], diag_idx[1]] = np.sqrt(stats.chi2.rvs(df=df))

        tril_idx = np.tril_indices(self._p, k=-1)
        size = (N_sample, self._p * (self._p - 1) // 2)
        A[:, tril_idx[0], tril_idx[1]] = stats.norm.rvs(size=size)

        W = self._L @ A
        X = W @ W.conj().transpose(0, 2, 1)
        return X
