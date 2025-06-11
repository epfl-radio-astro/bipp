# #############################################################################
# gram.py
# =======
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Gram-related operations and tools.
"""

import bipp.imot_tools.util.argcheck as chk
import numpy as np
import scipy.linalg as linalg

import bipp.core as core
import bipp.beamforming as beamforming
import bipp.instrument as instrument
import bipp.array as array
import bipp.numpy_compat as npc
import bipp.pybipp


class GramMatrix(array.LabeledMatrix):
    """
    Gram coefficients.

    Args
        data : array-like(complex)
            (N_beam, N_beam) Gram coefficients.
        beam_idx
            (N_beam,) index.
    """

    def __init__(self, data, beam_idx):
        data = npc.asarray(data)
        N_beam = len(beam_idx)

        if not chk.has_shape((N_beam, N_beam))(data):
            raise ValueError("Parameters[data, beam_idx] are not consistent.")

        if not np.allclose(data, data.conj().T):
            raise ValueError("Parameter[data] must be hermitian symmetric.")

        super().__init__(data, beam_idx, beam_idx)


class GramBlock(core.Block):
    """
    Compute Gram matrices.

    Args
        ctx: :py:class:`~bipp.Context`
            Bipp context.
    """

    def __init__(self, ctx):
        super().__init__()
        self._ctx = ctx

    def __call__(self, XYZ, W, wl):
        """
        Compute Gram matrix.

        Args
            XYZ : :py:class:`~bipp.phased_array.instrument.InstrumentGeometry`
                (N_antenna, 3) Cartesian antenna coordinates in any reference frame.
            W : :py:class:`~bipp.phased_array.beamforming.BeamWeights`
                (N_antenna, N_beam) synthesis beamweights.
            wl : float
                Wavelength [m] at which to compute the Gram.

        Returns
            :py:class:`~bipp.phased_array.gram.GramMatrix`
                (N_beam, N_beam) Gram matrix.
        """
        if not XYZ.is_consistent_with(W, axes=[0, 0]):
            raise ValueError("Parameters[XYZ, W] are inconsistent.")

        return GramMatrix(data=self.compute(XYZ.data, W.data, wl), beam_idx=W.index[1])

    def compute(self, XYZ, W, wl):
        """
        Compute Gram matrix as numpy array.

        Args
            XYZ : :py:class:`~bipp.phased_array.instrument.InstrumentGeometry`
                (N_antenna, 3) Cartesian antenna coordinates in any reference frame.
            W : :py:class:`~bipp.phased_array.beamforming.BeamWeights`
                (N_antenna, N_beam) synthesis beamweights.
            wl : float
                Wavelength [m] at which to compute the Gram.

        Returns
            :py:class:`~numpy.ndarray`
                (N_beam, N_beam) Gram matrix.
        """
        if self._ctx is not None:
            return bipp.pybipp.gram_matrix(
                self._ctx,
                np.array(XYZ.data, order="F"),
                np.array(W.data, order="F"),
                wl,
            )
        else:
            N_antenna = XYZ.shape[0]
            baseline = linalg.norm(
                XYZ.reshape(N_antenna, 1, 3) - XYZ.reshape(1, N_antenna, 3), axis=-1
            )

            G_1 = (4 * np.pi) * np.sinc((2 / wl) * baseline)
            G_2 = W.conj().T @ G_1 @ W
            return G_2
