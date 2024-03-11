r"""
Filters applied to eigenvalues
"""

import numpy as np

def apply_filter(f, D):
    if f == 'lsq':
        return D
    if f == 'std':
        D[D != 0] = 1
        return D
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

    def __init__(self, **kwargs):
        valid_filter = ['lsq', 'std', 'sqrt', 'inv', 'inv_sqrt']

        self._filter_intervals = []
        start_index = 0
        for name, intervals in kwargs.items():
            if not name in valid_filter:
                raise ValueError(f"Unknown filter: {name}")
            self._filter_intervals.append((name, start_index, intervals))
            start_index += intervals.shape[0]

        self._num_images = start_index

    def num_images(self):
        return self._num_images

    def num_filter(self):
        return len(self._filter_intervals)

    def get_filter_images(self, f, images):
        for name, start_index, intervals in self._filter_intervals:
            if name == f:
                if images.shape[0] < start_index + intervals.shape[0]:
                    raise ValueError(f"Expected {self._num_images} images, but got {images.shape[0]}.")
                return images[start_index : start_index + intervals.shape[0], :]

        raise ValueError(f"Filter {f} not found")

    def __call__(self, image_index, D):
        if image_index > self._num_images:
            raise ValueError("Image index out of bounds.")

        for name, start_index, intervals in self._filter_intervals:
            if image_index < start_index or image_index >= start_index + intervals.shape[0]:
                continue
            level = image_index - start_index

            D *= (D>=intervals[level, 0]) * (D<=intervals[level,1])
            return apply_filter(name, D)

        raise RuntimeError("Could not match image index to filter")

