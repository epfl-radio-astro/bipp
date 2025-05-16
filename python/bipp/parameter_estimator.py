# #############################################################################
# parameter_estimator.py
# ======================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

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
