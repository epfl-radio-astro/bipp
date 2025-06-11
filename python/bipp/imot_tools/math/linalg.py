# ##############################################################################
# linalg.py
# =========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# ##############################################################################

"""
Linear algebra routines.
"""

import numpy as np
import scipy.linalg as linalg

import bipp.imot_tools.util.argcheck as chk
import bipp.numpy_compat as npc


@chk.check(
    dict(
        A=chk.accept_any(chk.has_reals, chk.has_complex),
        B=chk.allow_None(chk.accept_any(chk.has_reals, chk.has_complex)),
        tau=chk.is_real,
        N=chk.allow_None(chk.is_integer),
    )
)
def eigh(A, B=None, tau=1, N=None):
    """
    Solve a generalized eigenvalue problem.

    Finds :math:`(D, V)`, solution of the generalized eigenvalue problem

    .. math::

       A V = B V D.

    This function is a wrapper around :py:func:`scipy.linalg.eigh` that adds energy truncation and
    extra output formats.

    Parameters
    ----------
    A : :py:class:`~numpy.ndarray`
        (M, M) hermitian matrix.
        If `A` is not positive-semidefinite (PSD), its negative spectrum is discarded.
    B : :py:class:`~numpy.ndarray`, optional
        (M, M) PSD hermitian matrix.
        If unspecified, `B` is assumed to be the identity matrix.
    tau : float, optional
        Normalized energy ratio. (Default: 1)
    N : int, optional
        Number of eigenpairs to output. (Default: K, the minimum number of leading eigenpairs that
        account for `tau` percent of the total energy.)

        * If `N` is smaller than K, then the trailing eigenpairs are dropped.
        * If `N` is greater that K, then the trailing eigenpairs are set to 0.

    Returns
    -------
    D : :py:class:`~numpy.ndarray`
        (N,) positive real-valued eigenvalues.

    V : :py:class:`~numpy.ndarray`
        (M, N) complex-valued eigenvectors.

        The N eigenpairs are sorted in decreasing eigenvalue order.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.math.linalg import eigh
       import scipy.linalg as linalg

       np.random.seed(0)

       def hermitian_array(N: int) -> np.ndarray:
           '''
           Construct a (N, N) Hermitian matrix.
           '''
           D = np.arange(N)
           Rmtx = np.random.randn(N,N) + 1j * np.random.randn(N, N)
           Q, _ = linalg.qr(Rmtx)

           A = (Q * D) @ Q.conj().T
           return A

       M = 4
       A = hermitian_array(M)
       B = hermitian_array(M) + 100 * np.eye(M)  # To guarantee PSD

    Let `A` and `B` be defined as below:

    .. doctest::

       M = 4
       A = hermitian_array(M)
       B = hermitian_array(M) + 100 * np.eye(M)  # To guarantee PSD

    Then different calls to :py:func:`~imot_tools.math.linalg.eigh` produce different results:

    * Get all positive eigenpairs:

    .. doctest::

       >>> D, V = eigh(A, B)
       >>> print(np.around(D, 4))  # The last term is small but positive.
       [0.0296 0.0198 0.0098 0.    ]

       >>> print(np.around(V, 4))
       [[-0.0621+0.0001j -0.0561+0.0005j -0.0262-0.0004j  0.0474+0.0005j]
        [ 0.0285+0.0041j -0.0413-0.0501j  0.0129-0.0209j -0.004 -0.0647j]
        [ 0.0583+0.0055j -0.0443+0.0033j  0.0069+0.0474j  0.0281+0.0371j]
        [ 0.0363+0.0209j  0.0006+0.0235j -0.029 -0.0736j  0.0321+0.0142j]]

    * Drop some trailing eigenpairs:

    .. doctest::

       >>> D, V = eigh(A, B, tau=0.8)
       >>> print(np.around(D, 4))
       [0.0296]

       >>> print(np.around(V, 4))
       [[-0.0621+0.0001j]
        [ 0.0285+0.0041j]
        [ 0.0583+0.0055j]
        [ 0.0363+0.0209j]]

    * Pad output to certain size:

    .. doctest::

       >>> D, V = eigh(A, B, tau=0.8, N=3)
       >>> print(np.around(D, 4))
       [0.0296 0.     0.    ]

       >>> print(np.around(V, 4))
       [[-0.0621+0.0001j  0.    +0.j      0.    +0.j    ]
        [ 0.0285+0.0041j  0.    +0.j      0.    +0.j    ]
        [ 0.0583+0.0055j  0.    +0.j      0.    +0.j    ]
        [ 0.0363+0.0209j  0.    +0.j      0.    +0.j    ]]
    """
    A = npc.asarray(A)
    M = len(A)
    if not (chk.has_shape([M, M])(A) and np.allclose(A, A.conj().T)):
        raise ValueError("Parameter[A] must be hermitian symmetric.")

    B = np.eye(M) if (B is None) else npc.asarray(B)
    if not (chk.has_shape([M, M])(B) and np.allclose(B, B.conj().T)):
        raise ValueError("Parameter[B] must be hermitian symmetric.")

    if not (0 < tau <= 1):
        raise ValueError("Parameter[tau] must be in [0, 1].")

    if (N is not None) and (N <= 0):
        raise ValueError(f"Parameter[N] must be a non-zero positive integer.")

    # A: drop negative spectrum.
    Ds, Vs = linalg.eigh(A)
    idx = Ds > 0
    Ds, Vs = Ds[idx], Vs[:, idx]
    A = (Vs * Ds) @ Vs.conj().T

    # A, B: generalized eigenvalue-decomposition.
    try:
        D, V = linalg.eigh(A, B)

        # Discard near-zero D due to numerical precision.
        idx = D > 0
        D, V = D[idx], V[:, idx]
        idx = np.argsort(D)[::-1]
        D, V = D[idx], V[:, idx]
    except linalg.LinAlgError:
        raise ValueError("Parameter[B] is not PSD.")

    # Energy selection / padding
    idx = np.clip(np.cumsum(D) / np.sum(D), 0, 1) <= tau
    D, V = D[idx], V[:, idx]
    if N is not None:
        M, K = V.shape
        if N - K <= 0:
            D, V = D[:N], V[:, :N]
        else:
            D = np.concatenate((D, np.zeros(N - K)), axis=0)
            V = np.concatenate((V, np.zeros((M, N - K))), axis=1)

    return D, V


@chk.check(
    dict(axis=chk.require_all(chk.has_reals, chk.has_shape((3,))), angle=chk.is_real)
)
def rot(axis, angle):
    """
    3D rotation matrix.

    Parameters
    ----------
    axis : :py:class:`~numpy.ndarray`
        (3,) rotation axis.
    angle : float
        signed rotation angle [rad].

    Returns
    -------
    R : :py:class:`~numpy.ndarray`
        (3, 3) rotation matrix.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.math.linalg import rot

    .. doctest::

       >>> R = rot([0, 0, 1], np.deg2rad(90))
       >>> np.around(R, 2)
       array([[ 0., -1.,  0.],
              [ 1.,  0.,  0.],
              [ 0.,  0.,  1.]])

       >>> R = rot([1, 0, 0], - 1)
       >>> np.around(R, 2)
       array([[ 1.  ,  0.  ,  0.  ],
              [ 0.  ,  0.54,  0.84],
              [ 0.  , -0.84,  0.54]])
    """
    axis = npc.asarray(axis)

    a, b, c = axis / linalg.norm(axis)
    ct, st = np.cos(angle), np.sin(angle)

    p00 = a**2 + (b**2 + c**2) * ct
    p11 = b**2 + (a**2 + c**2) * ct
    p22 = c**2 + (a**2 + b**2) * ct
    p01 = a * b * (1 - ct) - c * st
    p10 = a * b * (1 - ct) + c * st
    p12 = b * c * (1 - ct) - a * st
    p21 = b * c * (1 - ct) + a * st
    p20 = a * c * (1 - ct) - b * st
    p02 = a * c * (1 - ct) + b * st

    R = np.array([[p00, p01, p02], [p10, p11, p12], [p20, p21, p22]])
    return R


@chk.check("R", chk.require_all(chk.has_reals, chk.has_shape((3, 3))))
def z_rot2angle(R):
    """
    Determine rotation angle from Z-axis rotation matrix.

    Parameters
    ----------
    R : :py:class:`~numpy.ndarray`
        (3, 3) rotation matrix around the Z-axis.

    Returns
    -------
    angle : float
        Signed rotation angle [rad].

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.math.linalg import z_rot2angle

    .. doctest::

       >>> R = np.eye(3)
       >>> angle = z_rot2angle(R)
       >>> np.around(angle, 2)
       0.0

       >>> R = [[0, -1, 0],
       ...      [1,  0, 0],
       ...      [0,  0, 1]]
       >>> angle = z_rot2angle(R)
       >>> np.around(angle, 2)
       1.57
    """
    R = npc.asarray(R)

    if not np.allclose(R[[0, 1, 2, 2, 2], [2, 2, 2, 0, 1]], np.r_[0, 0, 1, 0, 0]):
        raise ValueError("Parameter[R] is not a rotation matrix around the Z-axis.")

    ct, st = np.clip([R[0, 0], R[1, 0]], -1, 1)
    if st >= 0:  # In quadrants I or II
        angle = np.arccos(ct)
    else:  # In quadrants III or IV
        angle = -np.arccos(ct)

    return angle
