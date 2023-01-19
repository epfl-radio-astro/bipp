# ##############################################################################
# special.py
# ==========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# ##############################################################################

"""
Special mathematical functions.
"""

import pathlib

import numpy as np
import pandas as pd
import pkg_resources as pkg
import scipy.special as special

import bipp.imot_tools.util.argcheck as chk


@chk.check("x", chk.is_real)
def jv_threshold(x):
    r"""
    Decay threshold of Bessel function :math:`J_{n}(x)`.

    Parameters
    ----------
    x : float

    Returns
    -------
    n : int
        Value of `n` in :math:`J_{n}(x)` past which :math:`J_{n}(x) \approx 0`.
    """
    rel_path = pathlib.Path("data", "math", "special", "jv_threshold.csv")
    abs_path = pkg.resource_filename("imot_tools", str(rel_path))

    data = pd.read_csv(abs_path).sort_values(by="x")
    x = np.abs(x)
    idx = int(np.digitize(x, bins=data["x"].values))
    if idx == 0:  # Below smallest known x.
        n = data["n_threshold"].iloc[0]
    else:
        if idx == len(data):  # Above largest known x.
            ratio = data["n_threshold"].iloc[-1] / data["x"].iloc[-1]
        else:
            ratio = data["n_threshold"].iloc[idx - 1] / data["x"].iloc[idx - 1]
        n = int(np.ceil(ratio * x))

    return n


@chk.check("x", chk.is_real)
def spherical_jn_threshold(x):
    r"""
    Decay threshold of spherical Bessel function :math:`j_{n}(x)`.

    Parameters
    ----------
    x : float

    Returns
    -------
    n : int
        Value of `n` in :math:`j_{n}(x)` past which :math:`j_{n}(x) \approx 0`.
    """
    rel_path = pathlib.Path("data", "math", "special", "spherical_jn_threshold.csv")
    abs_path = pkg.resource_filename("imot_tools", str(rel_path))

    data = pd.read_csv(abs_path).sort_values(by="x")
    x = np.abs(x)
    idx = int(np.digitize(x, bins=data["x"].values))
    if idx == 0:  # Below smallest known x.
        n = data["n_threshold"].iloc[0]
    else:
        if idx == len(data):  # Above largest known x.
            ratio = data["n_threshold"].iloc[-1] / data["x"].iloc[-1]
        else:
            ratio = data["n_threshold"].iloc[idx - 1] / data["x"].iloc[idx - 1]
        n = int(np.ceil(ratio * x))

    return n


@chk.check("x", chk.is_real)
def ive_threshold(x):
    r"""
    Decay threshold of the exponentially scaled Bessel function :math:`I_{n}^{e}(x) = I_{n}(x) e^{-|\Re{\{x\}}|}`.

    Parameters
    ----------
    x : float

    Returns
    -------
    n : int
        Value of `n` in :math:`I_{n}^{e}(x)` past which :math:`I_{n}^{e}(x) \approx 0`.
    """
    rel_path = pathlib.Path("data", "math", "special", "ive_threshold.csv")
    abs_path = pkg.resource_filename("imot_tools", str(rel_path))

    data = pd.read_csv(abs_path).sort_values(by="x")
    x = np.abs(x)
    idx = int(np.digitize(x, bins=data["x"].values))
    if idx == 0:  # Below smallest known x.
        n = data["n_threshold"].iloc[0]
    else:
        if idx == len(data):  # Above largest known x.
            ratio = data["n_threshold"].iloc[-1] / data["x"].iloc[-1]
        else:
            ratio = data["n_threshold"].iloc[idx - 1] / data["x"].iloc[idx - 1]
        n = int(np.ceil(ratio * x))

    return n


@chk.check(dict(x=chk.is_real, table_lookup=chk.is_boolean, epsilon=chk.is_real))
def spherical_jn_series_threshold(x, table_lookup=True, epsilon=1e-2):
    r"""
    Convergence threshold of series :math:`f_{n}(x) = \sum_{q = 0}^{n} (2 q + 1) j_{q}^{2}(x)`.

    Parameters
    ----------
    x : float
    table_lookup : bool
        Use pre-computed table (with `epsilon=1e-2`) to accelerate the search.
    epsilon : float
        Only used when `table_lookup` is :py:obj:`False`.

    Returns
    -------
    n : int
        Value of `n` in :math:`f_{n}(x)` past which :math:`f_{n}(x) \ge 1 - \epsilon`.
    """
    if not (0 < epsilon < 1):
        raise ValueError("Parameter[epsilon] must lie in (0, 1).")

    if table_lookup is True:
        rel_path = pathlib.Path(
            "data", "math", "special", "spherical_jn_series_threshold.csv"
        )
        abs_path = pkg.resource_filename("bipp.imot_tools", str(rel_path))

        data = pd.read_csv(abs_path).sort_values(by="x")
        x = np.abs(x)
        idx = int(np.digitize(x, bins=data["x"].values))
        if idx == 0:  # Below smallest known x.
            n = data["n_threshold"].iloc[0]
        else:
            if idx == len(data):  # Above largest known x.
                ratio = data["n_threshold"].iloc[-1] / data["x"].iloc[-1]
            else:
                ratio = data["n_threshold"].iloc[idx - 1] / data["x"].iloc[idx - 1]
            n = int(np.ceil(ratio * x))

        return n
    else:

        @chk.check(dict(n=chk.is_integer, x=chk.is_real))
        def series(n, x):
            q = np.arange(n)
            _2q1 = 2 * q + 1
            _sph = special.spherical_jn(q, x) ** 2

            return np.sum(_2q1 * _sph)

        n_opt = int(0.95 * x)
        while True:
            n_opt += 1
            if 1 - series(n_opt, x) < epsilon:
                return n_opt


@chk.check("x", chk.is_real)
def jv_series_threshold(x):
    r"""
    Convergence threshold of series :math:`f_{n}(x) = \sum_{q = -n}^{n} J_{q}^{2}(x)`.

    Parameters
    ----------
    x : float

    Returns
    -------
    n : int
        Value of `n` in :math:`f_{n}(x)` past which :math:`f_{n}(x) \ge 1 - \epsilon`.
    """
    rel_path = pathlib.Path("data", "math", "special", "jv_series_threshold.csv")
    abs_path = pkg.resource_filename("imot_tools", str(rel_path))

    data = pd.read_csv(abs_path).sort_values(by="x")
    x = np.abs(x)
    idx = int(np.digitize(x, bins=data["x"].values))
    if idx == 0:  # Below smallest known x.
        n = data["n_threshold"].iloc[0]
    else:
        if idx == len(data):  # Above largest known x.
            ratio = data["n_threshold"].iloc[-1] / data["x"].iloc[-1]
        else:
            ratio = data["n_threshold"].iloc[idx - 1] / data["x"].iloc[idx - 1]
        n = int(np.ceil(ratio * x))

    return n
