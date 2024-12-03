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

def centroid_to_intervals(centroid=None, min_pos_d=0.0, fne=True):
    r"""
    Convert centroid to invervals as required by VirtualVisibilitiesDataProcessingBlock.

    Args
        centroid: Optional[np.ndarray]
            (N_centroid) centroid values. If None, [0, max_float] is returned.
        min_pos_d: Optional[float]
            Optional lower bound for the first interval containing the smallest positive eigenvalues
            (i.e. the smallest of all (positive) eigenvalues considered when computing the centroids).
            Defaults to zero.
        fne: Optional[bool]
            Filter out negative eigenvalues (default is True to only keep positive ones).

    Returns
        intervals: np.ndarray
         (N_centroid, 2) Intervals matching the input with lower and upper bound.
    """
    if centroid is None or centroid.size == 0:
        return np.array([[0, np.finfo("f").max]])

    if centroid.size == 1:
        if fne:
            return np.array([[min_pos_d, np.finfo("f").max]])
        else:
            return np.array([[0, np.finfo("f").max],
                             [np.finfo("f").min, -np.finfo("f").tiny]])
    
    intervals = np.empty((centroid.size, 2))
    sorted_idx = np.argsort(centroid)
    sorted_centroid = centroid[sorted_idx]
    for i in range(centroid.size):
        idx = sorted_idx[i]
        if idx == 0:
            intervals[i, 0] = min_pos_d if fne else 0
        else:
            intervals[i, 0] = (sorted_centroid[idx] + sorted_centroid[idx - 1]) / 2

        if idx == centroid.size - 1:
            intervals[i, 1] = np.finfo("f").max
        else:
            intervals[i, 1] = (sorted_centroid[idx] + sorted_centroid[idx + 1]) / 2

    if not fne:
        intervals = np.append(intervals, [[np.finfo("f").min, -np.finfo("f").tiny]], axis=0)

    return intervals


class ParameterEstimator:
    """
    Parameter estimator for computing intensity fields.

    Args
        N_level : int
            Number of clustered energy levels to output.
        sigma : float
            Normalized energy ratio for fPCA decomposition.
    """

    def __init__(self, N_level, sigma, fne: bool=True):
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
        self._inferred = False
        self._fne = fne

    def collect(self, D):
        """
        Ingest data to internal queue for inference.

        Args
            S : :py:class:`~bipp.phased_array.data_gen.statistics.VisibilityMatrix`
                (N_beam, N_beam) visibility matrix.
            G : :py:class:`~bipp.phased_array.bipp.gram.GramMatrix`
                (N_beam, N_beam) gram matrix.
        """
        D = D[D > 0.0]
        D = D[np.argsort(D)[::-1]]
        if self._fne:
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

        min_pos_d = np.min(D_all) if self._fne else 0
        
        self._intervals = centroid_to_intervals(cluster_centroid, min_pos_d, self._fne)
        return self._intervals
