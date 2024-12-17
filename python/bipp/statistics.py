# #############################################################################
# statistics.py
# =============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Visibility generation utilities.

Due to the high data-rates emanating from antennas, raw antenna time-series are rarely archived.
Instead, signals from different antennas are correlated together to form *visibility* matrices.
"""

import bipp.imot_tools.math.stat as stat
import bipp.imot_tools.util.argcheck as chk
import numpy as np

import bipp.core as core
import bipp.beamforming as beamforming
import bipp.instrument as instrument
import bipp.source as sky
import bipp.array as array


class VisibilityMatrix(array.LabeledMatrix):
    """
    Visibility coefficients.

    Args
        data : array-like(real, complex)
            (N_beam, N_beam) visibility coefficients.
        beam_idx
            (N_beam,) index.
    """

    def __init__(self, data, beam_idx):
        data = np.asarray(data)
        N_beam = len(beam_idx)

        if not chk.has_shape((N_beam, N_beam))(data):
            raise ValueError("Parameters[data, beam_idx] are not consistent.")

        if not np.allclose(data, data.conj().T):
            raise ValueError("Parameter[data] must be hermitian symmetric.")

        # Always flag autocorrelation visibilities
        np.fill_diagonal(data, 0)
        
        super().__init__(data, beam_idx, beam_idx)


class VisibilityGeneratorBlock(core.Block):
    """
    Generate synthetic visibility matrices.

    Args
        sky_model : :py:class:`~bipp.phased_array.data_gen.source.SkyEmission`
            Source model from which to generate data.
        T : float
            Integration time [s].
        fs : int
            Sampling rate [samples/s].
        SNR : float
            Signal-to-Noise-Ratio (dB).
    """

    def __init__(self, sky_model, T, fs, SNR):
        super().__init__()
        self._N_sample = int(T * fs) + 1
        self._SNR = 10 ** (SNR / 10)
        self._sky_model = sky_model

    def __call__(self, XYZ, W, wl):
        """
        Compute visibility matrix.

        Args
            XYZ : :py:class:`~bipp.phased_array.instrument.InstrumentGeometry`
                (N_antenna, 3) ICRS instrument geometry.
            W : :py:class:`~bipp.phased_array.beamforming.BeamWeights`
                (N_antenna, N_beam) synthesis beamweights.
            wl : float
                Wavelength [m] at which to generate visibilities.

        Returns
            :py:class:`~bipp.phased_array.data_gen.statistics.VisibilityMatrix`
                (N_beam, N_beam) visibility matrix.
        """
        if not XYZ.is_consistent_with(W, axes=[0, 0]):
            raise ValueError("Parameters[XYZ, W] are inconsistent.")

        A = np.exp((1j * 2 * np.pi / wl) * (self._sky_model.xyz @ XYZ.data.T))
        S_sky = (W.data.conj().T @ (A.conj().T * self._sky_model.intensity)) @ (
            A @ W.data
        )

        noise_var = np.sum(self._sky_model.intensity) / (2 * self._SNR)
        S_noise = W.data.conj().T @ (noise_var * W.data)

        wishart = stat.Wishart(V=S_sky + S_noise, n=self._N_sample)
        S = wishart()[0] / self._N_sample
        return VisibilityMatrix(data=S, beam_idx=W.index[1])
