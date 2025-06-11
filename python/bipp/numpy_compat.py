import numpy as np

def asarray(obj, dtype=None):
    """
    Replacement of np.array(obj, copy=False), compatible with NumPy 1.x and 2.x.
    If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior change in NumPy 1.x).
    For more details, see https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.

    Args:
        obj: Input object.
        dtype: Optional dtype.

    Returns:
        A NumPy array, avoiding unnecessary copies.
    """

    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        return np.asarray(obj, dtype=dtype)
    else:
        return np.array(obj, copy=False, dtype=dtype)
