r"""
Filters applied to eigenvalues
"""

import numpy as np

def apply_filter(f, D):
    if f == 'lsq':
        return D
    if f == 'std':
        return np.sign(D)
    if f == 'sqrt':
        sign = np.sign(D)
        D = sign * np.sqrt(sign * D)
        return D
    if f == 'inv_sqrt':
        sign = np.sign(D)
        D = sign * np.sqrt(sign * D)
        non_zero = D != 0
        D[non_zero] = 1.0 / D[non_zero]
        return D
    if f == 'inv':
        non_zero = D != 0
        D[non_zero] = 1.0 / D[non_zero]
        return D
    raise ValueError(f"Unknown filter: {f}")

class Filter:
    """
    Filter

    Args
        filter_intervals : dictionary(string : :py:class:`~numpy.ndarray`)
            (N_filter, (N_level,2)) Dictionary, mapping filter to individual intervals for eigenvalue selection.
    """

    def __init__(self, filter_name, lower_bound, upper_bound):
        valid_filter = ['lsq', 'std', 'sqrt', 'inv', 'inv_sqrt']

        if not filter_name in valid_filter:
            raise ValueError(f"Unknown filter: {filter_name}")

        self.filter_name = filter_name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound



    def __call__(self, D):
        D *= (D>=self.lower_bound) * (D<=self.upper_bound)
        return apply_filter(self.filter_name, D)

        raise RuntimeError("Could not match image index to filter")

