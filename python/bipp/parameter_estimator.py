# #############################################################################
# parameter_estimator.py
# ======================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

r"""
Parameter estimators.

Bipp field synthesizers output :math:`N_{\text{beam}}` energy levels, with :math:`N_{\text{beam}}`
being the height of the visibility/Gram matrices :math:`\Sigma, G`.
We are often not interested in such fined-grained energy decompositions but would rather have 4-5
well-separated energy levels as output.
This is accomplished by clustering energy levels together during the aggregation stage.

As the energy scale depends on the visibilities, it is preferable to infer the cluster centroids
(and any other parameters of interest) by scanning a portion of the data stream.
Subclasses of :py:class:`~bipp.phased_array.bipp.parameter_estimator.ParameterEstimator` are
specifically tailored for such tasks.
"""

import bipp.imot_tools.math.linalg as pylinalg
import bipp.imot_tools.util.argcheck as chk
import numpy as np
import sklearn.cluster as skcl

import bipp.statistics as vis
import bipp.gram as gr
import bipp.filter
import bipp.pybipp


def centroid_to_intervals(centroid=None, d_min=0.0):
    r"""
    Convert centroid to invervals as required by VirtualVisibilitiesDataProcessingBlock.

    Args
        centroid: Optional[np.ndarray]
            (N_centroid) centroid values. If None, [0, max_float] is returned.
        d_min: Optional[float]
            Optional lower bound of first interval containing the smallest eigenvalues (i.e. the
            smallest of all eigenvalues considered when computing the centroids).
            Defaults to zero.
    Returns
        intervals: np.ndarray
         (N_centroid, 2) Intervals matching the input with lower and upper bound.
    """
    if centroid is None or centroid.size <= 1:
        return np.array([[0, np.finfo("f").max]])
    intervals = np.empty((centroid.size, 2))
    sorted_idx = np.argsort(centroid)
    sorted_centroid = centroid[sorted_idx]
    for i in range(centroid.size):
        idx = sorted_idx[i]
        if idx == 0:
            intervals[i, 0] = d_min
        else:
            intervals[i, 0] = (sorted_centroid[idx] + sorted_centroid[idx - 1]) / 2

        if idx == centroid.size - 1:
            intervals[i, 1] = np.finfo("f").max
        else:
            intervals[i, 1] = (sorted_centroid[idx] + sorted_centroid[idx + 1]) / 2

    return intervals


class ParameterEstimator:
    """
    Parameter estimator for computing intensity fields.

    Args
        N_level : int
            Number of clustered energy levels to output.
        sigma : float
            Normalized energy ratio for fPCA decomposition.
        ctx: :py:class:`~bipp.Context`
            Bipp context. If provided, will use bipp module for computation.
    """

    def __init__(self, N_level, sigma, ctx):
        super().__init__()

        if N_level <= 0:
            raise ValueError("Parameter[N_level] must be positive.")
        self._N_level = N_level

        if not (0 < sigma <= 1):
            raise ValueError("Parameter[sigma] must lie in (0,1].")
        self._sigma = sigma

        # Collected data.
        self._intervals = []
        self._d_all = []
        self._ctx = ctx
        self._inferred = False

    def collect(self, wl, S, W, XYZ):
        """
        Ingest data to internal queue for inference.

        Args
            S : :py:class:`~bipp.phased_array.data_gen.statistics.VisibilityMatrix`
                (N_beam, N_beam) visibility matrix.
            G : :py:class:`~bipp.phased_array.bipp.gram.GramMatrix`
                (N_beam, N_beam) gram matrix.
        """

        D =  bipp.pybipp.eigh(self._ctx, wl,S, W, XYZ)
        D = D[D > 0.0]
        D = D[np.argsort(D)[::-1]]
        idx = np.clip(np.cumsum(D) / np.sum(D), 0, 1) <= self._sigma
        D = D[idx]
        self._d_all.append(D)
        self._inferred = False

    def num_level(self):
        if not self._inferred:
            self.infer_parameters()
        if len(self._intervals) == 0:
            return 0
        return self._intervals.shape[0]

    def infer_parameters(self):
        """
        Estimate parameters given ingested data.

        cluster_intervals : :py:class:`~numpy.ndarray`
            (N_level,2) intensity field intervals to select eigenvalues for each level.
        """
        if len(self._d_all) == 0:
            return 0, np.empty((0, 2))

        D_all = np.concatenate(self._d_all)
        kmeans = skcl.KMeans(n_clusters=self._N_level).fit(np.log(D_all).reshape(-1, 1))

        cluster_centroid = np.sort(np.exp(kmeans.cluster_centers_)[:, 0])[::-1]

        self._inferred = True
        self._intervals = centroid_to_intervals(cluster_centroid, np.min(D_all))
        return self._intervals
