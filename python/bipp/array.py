# #############################################################################
# array.py
# ========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Tools and utilities for manipulating arrays.
"""

import bipp.imot_tools.util.argcheck as chk
import numpy as np
import pandas as pd
import scipy.sparse as sparse


class LabeledMatrix:
    """
    2D arrays with additional indexing data attached to each axis.

    Args
        data : array-like
            (N, M) dataset (any type). Sparse CSR/CSC matrices are also accepted.
        row_idx : :py:class:`~pandas.Index`
            Row index.
        col_idx : :py:class:`~pandas.Index`
            Column index.
    """

    def __init__(self, data, row_idx, col_idx):
        if chk.is_instance(sparse.spmatrix)(data):
            self.__data = data

            if not (
                sparse.isspmatrix_csc(self.__data) or sparse.isspmatrix_csr(self.__data)
            ):
                raise ValueError("Parameter[data] must be CSC/CSR-ordered.")
        else:
            self.__data = np.array(data, copy=False)

            if self.__data.ndim != 2:
                raise ValueError("Parameter[data] must be 2D.")

        N, M = self.__data.shape
        N_row, N_col = len(row_idx), len(col_idx)
        if N_row != N:
            raise ValueError(
                f"Parameter[row_idx] contains {N_row} entries, "
                f"but Parameter[data] expected {N}."
            )
        if N_col != M:
            raise ValueError(
                f"Parameter[col_idx] contains {N_col} entries, "
                f"but Parameter[data] expected {M}."
            )
        self.__index = row_idx.copy(), col_idx.copy()

    @property
    def data(self):
        """
        Returns
            :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`
                (N, M) dataset.
        """
        return self.__data

    @property
    def index(self):
        """
        Returns
            row_idx : :py:class:`~pandas.Index`
                (N,) row index.

            col_idx : :py:class:`~pandas.Index`
                (M,) column index.
        """
        return self.__index

    @property
    def shape(self):
        """
        Returns
            tuple(int)
                (N_row, N_col) shape information.
        """
        return self.__data.shape

    def __str__(self):
        return self.__data.__str__()

    def __repr__(self):
        return self.__data.__repr__()

    def is_consistent_with(self, lmtx, axes):
        """
        Test matrices for consistency along directions.

        Two labeled matrices are considered consistent if their indexes match along the specified
        dimensions.

        Args
            lmtx : :py:class:`~pypeline.util.array.LabeledMatrix`
                Matrix to compare with.
            axes : tuple(int)
                (2,) tuple with dimensions along which indices of `self` and `lmtx` must match.

        Returns
            bool
                True if axes are consistent.

        """
        if not chk.is_instance(LabeledMatrix)(lmtx):
            raise ValueError("Parameter[lmtx] must be a LabeledMatrix.")

        axes = np.array(axes, copy=False)
        if not np.all((axes == 0) | (axes == 1)):
            raise ValueError("Parameter[axes] can only contain {0, 1}.")

        idxA = self.index[axes[0]]
        idxB = lmtx.index[axes[1]]

        if tuple(idxA.names) == tuple(idxB.names):
            if len(idxA) == len(idxB):
                if np.all(idxA == idxB):
                    return True

        return False
