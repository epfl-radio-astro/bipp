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
import bipp.pybipp


def centroid_to_intervals(centroid):
    r"""
    Convert centroid to invervals as required by VirtualVisibilitiesDataProcessingBlock.

    Args
        centroid: Optional[np.ndarray]
            (N_centroid) centroid values. If None, [0, max_float] is returned.

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
            intervals[i, 0] = 0
        else:
            intervals[i, 0] = (sorted_centroid[idx] + sorted_centroid[idx - 1]) / 2

        if idx == centroid.size - 1:
            intervals[i, 1] = np.finfo("f").max
        else:
            intervals[i, 1] = (sorted_centroid[idx] + sorted_centroid[idx + 1]) / 2

    return intervals


class ParameterEstimator:
    """
    Top-level public interface of Bipp parameter estimators.
    """

    def __init__(self):
        super().__init__()

    def collect(self, *args, **kwargs):
        """
        Ingest data to internal queue for inference.

        Args
            \*args
                Positional arguments.
            \*\*kwargs
                Keyword arguments.
        """
        raise NotImplementedError

    def infer_parameters(self):
        """
        Estimate parameters given ingested data.

        Returns
            tuple
                Parameters as defined by subclasses.
        """
        raise NotImplementedError


class IntensityFieldParameterEstimator(ParameterEstimator):
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
        self._visibilities = []
        self._grams = []
        self._ctx = ctx

    def collect(self, S, G):
        """
        Ingest data to internal queue for inference.

        Args
            S : :py:class:`~bipp.phased_array.data_gen.statistics.VisibilityMatrix`
                (N_beam, N_beam) visibility matrix.
            G : :py:class:`~bipp.phased_array.bipp.gram.GramMatrix`
                (N_beam, N_beam) gram matrix.
        """
        if not S.is_consistent_with(G, axes=[0, 0]):
            raise ValueError("Parameters[S, G] are inconsistent.")

        self._visibilities.append(S)
        self._grams.append(G)

    def infer_parameters(self, fne: bool=True):
        """
        Estimate parameters given ingested data.
        Args
            fne : bool
                Filter negative eigenvalues (default is True to only keep positive ones)
        Returns
            N_eig : int
                Number of eigenpairs to use.

        cluster_intervals : :py:class:`~numpy.ndarray`
            (N_level,2) intensity field cluster intervals.
        """
        N_data = len(self._visibilities)
        N_beam = N_eig_max = self._visibilities[0].shape[0]

        if self._N_level > N_beam:
            raise ValueError(
                f"Initialization parameter N_level (set to {self._N_level}) cannot exceed the number of beams ({N_beam})."
            )

        D_all = np.zeros((N_data, N_eig_max))
        D_all_neg = np.zeros((N_data, N_eig_max))
        
        for i, (S, G) in enumerate(zip(self._visibilities, self._grams)):
            # Remove broken BEAM_IDs
            broken_row_id = np.flatnonzero(np.isclose(np.sum(S.data, axis=0), 0))
            working_row_id = list(set(np.arange(N_beam)) - set(broken_row_id))
            idx = np.ix_(working_row_id, working_row_id)
            S, G = S.data[idx], G.data[idx]

            # Functional PCA
            if not np.allclose(S, 0):
                _, D, _ = bipp.pybipp.eigh(self._ctx, S.data.shape[0], S.data, G.data)
                # only consider positive eigenvalues, since we apply the log function for clustering
                N = D[D <= 0.0]
                D = D[D > 0.0]
                if fne:
                    D = D[np.argsort(D)[::-1]]
                    idx = np.clip(np.cumsum(D) / np.sum(D), 0, 1) <= self._sigma
                    D = D[idx]
                    D = D[np.argsort(D)]
                D_all[i, : len(D)] = D                    
                D_all_neg[i, : len(N)] = N
                
        D_all = D_all[D_all.nonzero()]
        D_all_neg = D_all_neg[D_all_neg.nonzero()]
        
        kmeans = skcl.KMeans(n_clusters=self._N_level).fit(np.log(D_all).reshape(-1, 1))

        # For extremely small telescopes or datasets that are mostly 'broken', we can have (N_eig < N_level).
        # In this case we have two options: (N_level = N_eig) or (N_eig = N_level).
        # In the former case, the user's choice of N_level is not respected and subsequent code written by the user
        # could break due to a false assumption. In the latter case, we modify N_eig to match the user's choice.
        # This has the disadvantage of increasing the computational load of Bipp, but as the N_eig energy levels
        # are clustered together anyway, the trailing energy levels will be (close to) all-0 and can be discarded
        # on inspection.

        cluster_centroids = np.sort(np.exp(kmeans.cluster_centers_)[:, 0])[::-1]
        intervals = centroid_to_intervals(cluster_centroids)
        
        if fne:
            N_eig = max(int(np.ceil(len(D_all) / N_data)), self._N_level)
        else:
            N_eig = max(int(np.ceil((len(D_all) + len(D_all_neg)) / N_data)), self._N_level)
            intervals = np.append(intervals, [[np.finfo("f").min, 0]], axis=0)

        return N_eig, intervals

    
class SensitivityFieldParameterEstimator(ParameterEstimator):
    """
    Parameter estimator for computing sensitivity fields.

    Args
        sigma : float
            Normalized energy ratio for fPCA decomposition.
        ctx: :py:class:`~bipp.Context`
            Bipp context. If provided, will use bipp module for computation.
    """

    def __init__(self, sigma, ctx):
        super().__init__()

        if not (0 < sigma <= 1):
            raise ValueError("Parameter[sigma] must lie in (0,1].")
        self._sigma = sigma

        # Collected data.
        self._grams = []
        self._ctx = ctx

    def collect(self, G):
        """
        Ingest data to internal queue for inference.

        Args
            G : :py:class:`~bipp.phased_array.bipp.gram.GramMatrix`
                (N_beam, N_beam) gram matrix.
        """
        self._grams.append(G)

    def infer_parameters(self):
        """
        Estimate parameters given ingested data.

        Returns
            N_eig : int
                Number of eigenpairs to use.
        """
        N_data = len(self._grams)
        N_beam = N_eig_max = self._grams[0].shape[0]

        D_all = np.zeros((N_data, N_eig_max))
        for i, G in enumerate(self._grams):
            # Functional PCA
            _, D, _ = bipp.pybipp.eigh(self._ctx, G.data.shape[0], G.data)
            idx = np.clip(np.cumsum(D) / np.sum(D), 0, 1) <= self._sigma
            D = D[idx]
            D_all[i, : len(D)] = D

        D_all = D_all[D_all.nonzero()]

        N_eig = int(np.ceil(len(D_all) / N_data))
        return N_eig
