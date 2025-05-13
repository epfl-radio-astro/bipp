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


def centroid_to_intervals(centroid=None, d_min=0.0, d_max=np.finfo("f").max):
    r"""
    Convert centroid to invervals as required by VirtualVisibilitiesDataProcessingBlock.

    Args
        centroid: Optional[np.ndarray]
            (N_centroid) centroid values. If None, [0, max_float] is returned.
        d_min: Optional[float]
            Optional lower bound.
        d_max: Optional[float]
            Optional upper bound.

    Returns
        intervals: np.ndarray
         (N_centroid, 2) Intervals matching the input with lower and upper bound.
    """
    if centroid is None or centroid.size <= 1:
        return np.array([[d_min, d_max]])

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
            intervals[i, 1] = d_max
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
    """

    def __init__(self, N_level, sigma, fne: bool = True):
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
        #  if self._fne:
        #      idx = np.clip(np.cumsum(D) / np.sum(D), 0, 1) <= self._sigma
        #      D = D[idx]
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

        d_min = np.min(D_all) if self._fne else 0

        self._intervals = centroid_to_intervals(cluster_centroid, d_min)
        return self._intervals


def infer_intervals(N_level, sigma, cluster_func, d_min, d_max, d_all):
    """
    Infer intervals to partition eigenvalues into.

    Args
        N_level: int
            Number of intervals / levels to partition into.
        sigma: float
            Fraction of lowest eigenvalues to use for clustering.
            For example 0.95 implies that the 95% lowest eigenvalues will be used for clustering.
        cluster_func: str
            Modifier function to use for clustering. Can be 'none' or 'log'.
        d_min: float
            Lower bound.
        d_max: float
            Upper bound.

    Returns
        intervals: np.ndarray
    """

    if cluster_func == "log" and d_min < 0:
        raise ValueError(
            f'Cluster function "log" requires non-negative minimum eigenvalue restriction. Got {d_min}'
        )

    d_all = np.array(d_all).flatten()
    d_all = d_all[d_all != 0.0]

    if sigma < 1 and sigma >= 0:
        n_remove = int((1 - sigma) * d_all.shape[0])
        if n_remove <= d_all.shape[0]:
            d_all = np.sort(d_all)
            d_all = d_all[0:-n_remove]

    d_all = d_all[d_all >= d_min]
    d_all = d_all[d_all <= d_max]

    d_all = d_all.reshape(-1, 1)

    seed = 42
    if cluster_func == "log":
        kmeans = skcl.KMeans(n_clusters=N_level, random_state=seed).fit(np.log(d_all))
        cluster_centroid = np.sort(np.exp(kmeans.cluster_centers_)[:, 0])[::-1]
    elif cluster_func == "none":
        kmeans = skcl.KMeans(n_clusters=N_level, random_state=seed).fit(d_all)
        cluster_centroid = np.sort(kmeans.cluster_centers_[:, 0])[::-1]
    else:
        raise ValueError(f'Unknown cluster function "{cluster_func}".')

    return centroid_to_intervals(cluster_centroid, d_min, d_max)
