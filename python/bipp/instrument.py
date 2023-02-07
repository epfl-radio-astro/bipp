# #############################################################################
# instrument.py
# =============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Instrument-related operations.

Phased-arrays are collections of receiving elements (i.e., antennas or microphones) that sense a
random field around them.
These instruments are characterized by the properties of their receiving elements, such as position,
sensitivity, etc.

Only positional information is modeled at the moment, and can be accessed through 2 objects:

 * :py:class:`~bipp.phased_array.instrument.InstrumentGeometryBlock` : compute positional
   information.
 * :py:class:`~bipp.phased_array.instrument.InstrumentGeometry` : container for positional
   information.
"""

import pathlib

import astropy.coordinates as coord
import astropy.time as time
import bipp.imot_tools.math.linalg as pylinalg
import bipp.imot_tools.math.special as sp
import bipp.imot_tools.math.sphere.transform as transform
import bipp.imot_tools.util.argcheck as chk
import numpy as np
import pandas as pd
import pkg_resources as pkg
import scipy.linalg as linalg

import bipp.core as core
import bipp.array as array
import bipp.frame as frame
import typing as typ


def is_antenna_index(x):
    """
    Return :py:obj:`True` if `x` is a valid antenna index.

    A valid antenna index is a :py:class:`~pandas.MultiIndex` having columns

    * STATION_ID : int
    * ANTENNA_ID : int
    """
    if chk.is_instance(pd.MultiIndex)(x):
        col_names = ("STATION_ID", "ANTENNA_ID")
        if tuple(x.names) == col_names:
            for col in col_names:
                ids = x.get_level_values(col).values
                if not chk.has_integers(ids):
                    return False
            return True

    return False


def _as_InstrumentGeometry(df):
    XYZ = InstrumentGeometry(xyz=df.values, ant_idx=df.index)
    return XYZ

def _check_requested_geometry(N_station_conf, N_station, instrument_configuration_file):
    if N_station > N_station_conf:
        msg = (f"Parameter[N_station] set to {N_station} exceeds the " +
               f"number of stations ({N_station_conf}) in instrument\'s configuration file: " +
               f"{instrument_configuration_file}")
        raise ValueError(msg)


class InstrumentGeometry(array.LabeledMatrix):
    """
    Position of antennas in a particular reference frame.

    Args
        xyz : :py:class:`~numpy.ndarray`
            (N_antenna, 3) Cartesian coordinates.
        ant_idx : :py:class:`~pandas.MultiIndex`
            (N_antenna,) index.
    """

    def __init__(self, xyz, ant_idx):
        xyz = np.array(xyz, copy=False)
        N_antenna = len(xyz)
        if not chk.has_shape((N_antenna, 3))(xyz):
            raise ValueError("Parameter[xyz] must be a (N_antenna, 3) array.")

        N_idx = len(ant_idx)
        if N_idx != N_antenna:
            raise ValueError(
                "Parameter[xyz] and Parameter[ant_idx] are not compatible."
            )

        col_idx = pd.Index(["X", "Y", "Z"], name="COORDINATE")
        super().__init__(xyz, ant_idx, col_idx)

    def as_frame(self):
        """
        Returns
            :py:class:`~pandas.DataFrame`
                (N_antenna, 3) view of coordinates.
        """
        df = pd.DataFrame(
            data=self.data, index=self.index[0], columns=self.index[1].values
        )
        return df


class InstrumentGeometryBlock(core.Block):
    """
    Compute antenna positions.

    Args
         XYZ : :py:class:`~bipp.phased_array.instrument.InstrumentGeometry`
             Instrument geometry.
         N_station : int
             Number of stations to use. (Default = all)

         Sometimes only a subset of an instrument's stations are desired.
         Setting `N_station` limits the number of stations to those that appear first in `XYZ`
         when sorted by STATION_ID.
    """

    def __init__(self, XYZ, N_station=None):
        super().__init__()
        if N_station is not None:
            if N_station < 1:
                raise ValueError("Parameter[N_station] must be positive.")

        self._layout = XYZ.as_frame()

        if N_station is not None:
            stations = np.unique(XYZ.index[0].get_level_values("STATION_ID"))[
                :N_station
            ]
            self._layout = self._layout.loc[stations]

    def __call__(self, *args, **kwargs):
        """
        Determine instrument antenna positions.

        Args
            \*args
                Positional arguments.
            \*\*kwargs
                Keyword arguments.
        Returns
            :py:class:`~bipp.phased_array.instrument.InstrumentGeometry`
                (N_antenna, 3) instrument geometry.
        """
        raise NotImplementedError

    def nyquist_rate(self, wl):
        """
        Order of imageable complex plane-waves.

        Args
            wl : float
                Wavelength [m] of observations.

        Returns
            N : int
                Maximum order of complex plane waves that can be imaged by the instrument.
        """
        XYZ = self._layout.values
        baseline = linalg.norm(XYZ[:, np.newaxis, :] - XYZ[np.newaxis, :, :], axis=-1)

        N = sp.spherical_jn_series_threshold((2 * np.pi / wl) * baseline.max())
        return N


class EarthBoundInstrumentGeometryBlock(InstrumentGeometryBlock):
    """
    Sub-class specialized in instruments that move with the Earth, such as radio telescopes.

    Args
        XYZ : :py:class:`~bipp.phased_array.instrument.InstrumentGeometry`
            ITRS instrument geometry.
        N_station : int
            Number of stations to use. (Default = all)

        Sometimes only a subset of an instrument’s stations are desired.
        Setting `N_station` limits the number of stations to those that appear first in `XYZ`
        when sorted by STATION_ID.
    """

    def __init__(self, XYZ, N_station=None):
        super().__init__(XYZ, N_station)

    def __call__(self, time):
        """
        Determine instrument antenna positions in ICRS.

        Args
            time : :py:class:`~astropy.time.Time`
                Moment at which the coordinates are wanted.

        Returns
            :py:class:`~bipp.phased_array.instrument.InstrumentGeometry`
                (N_antenna, 3) ICRS instrument geometry.
        """
        layout = self._layout.loc[:, ["X", "Y", "Z"]].values.T
        r = linalg.norm(layout, axis=0)

        itrs_layout = coord.CartesianRepresentation(layout)
        itrs_position = coord.SkyCoord(itrs_layout, obstime=time, frame="itrs")
        icrs_position = r * (itrs_position.transform_to("icrs").cartesian.xyz)
        icrs_layout = pd.DataFrame(
            data=icrs_position.T, index=self._layout.index, columns=("X", "Y", "Z")
        )
        return _as_InstrumentGeometry(icrs_layout)

    def baselines(
        self,
        t: time.Time,
        uvw: bool = False,
        field_center: typ.Optional[coord.SkyCoord] = None,
    ) -> np.ndarray:
        r"""
        Baselines of the instrument at a given time.

        Args
            t: astropy.time.time
                Time at which the coordinates are wanted.
            uvw: bool
                If ``True``, the baselines coordinates are expressed in the local UVW frame, attached to the center of the FoV.
                If ``False``, the baseline coordinates are expressed in the ICRS frame.
            field_center: Optional[astropy.coordinates.SkyCoord]
                If ``uvw=True`` this argument specifies the center of the FoV used to define the local UVW frame.

        Returns
            baselines: np.ndarray
                (N_antenna, N_antenna, 3) baselines coordinates.
        """
        XYZ = self.__call__(t).data
        if uvw:
            if field_center is None:
                raise ValueError(
                    "Please provide a field_center for uvw coordinates conversion."
                )
            uvw_frame = frame.uvw_basis(field_center)
            UVW = (uvw_frame.transpose() @ XYZ.transpose()).transpose()
            baselines = UVW[:, None, :] - UVW[None, ...]
        else:
            baselines = XYZ[:, None, :] - XYZ[None, ...]
        return baselines

    def icrs2bfsf_rot(self, obs_start, obs_end):
        """
        Rotation matrix from ICRS to the local *Bipp FastSynthesis Frame* (BFSF).

        Args
            obs_start : :py:class:`~astropy.time.Time`
                Start of the observation period.
            obs_end : :py:class:`~astropy.time.Time`
                End of the observation period.

        Returns
            R : :py:class:`~numpy.ndarray`
                (3, 3) ICRS -> BFSF rotation matrix.
        """
        if obs_start > obs_end:
            raise ValueError("Parameter[obs_start] must precede Parameter[obs_end].")

        # Find the position of the antennas at several time-instants
        # during `period`.
        N_interval = 20
        sampling_times = obs_start + (
            (obs_end - obs_start) / (N_interval - 1)
        ) * np.arange(N_interval)
        icrs_layouts = [self.__call__(t).data for t in sampling_times]
        icrs_layouts = np.stack(icrs_layouts, axis=1)  # (N_antenna, N_time, 3)

        # For each antenna `i`, find a normal vector `n_{i}` to the rotation plane.
        N_antenna = len(icrs_layouts)
        abc = np.zeros((N_antenna, 3))
        for i, xyz in enumerate(icrs_layouts):
            a = np.concatenate([xyz[:, :2], np.ones((len(xyz), 1))], axis=-1)
            b = xyz[:, 2]
            coeffs, *_ = linalg.lstsq(a, b)
            abc[i] = coeffs

        # Average vectors `n_{i}` to obtain the global normal vector `n`.
        # Construct rotation matrix R using 3 basis vectors Ex, Ey, Ez.
        #    Ez must be co-linear to the plane's normal vector.
        a, b, c = np.mean(abc, axis=0)
        z_ax = np.r_[a, b, -1]
        y_ax = np.r_[b, -a, 0]
        x_ax = np.cross(z_ax, y_ax)
        R = np.stack([x_ax, y_ax, z_ax], axis=0)
        R /= linalg.norm(R, axis=1, keepdims=True)
        return R

    def bfsf_kernel_bandwidth(self, wl, obs_start, obs_end):
        r"""
        Bandwidth of :math:`2 \pi`-periodic complex plane-wave kernel in BFSF coordinates.

        Args
            wl : float
                Wavelength [m] of observations.
            obs_start : :py:class:`~astropy.time.Time`
                Start of the observation period.
            obs_end : :py:class:`~astropy.time.Time`
                End of the observation period.

        Returns
            N_FS : int
                Kernel bandwidth when evaluated in the BFSF reference frame.
        """
        R = self.icrs2bfsf_rot(obs_start, obs_end)
        obs_mid = obs_start + (obs_end - obs_start) / 2

        icrs_XYZ = self.__call__(obs_mid).data
        bfsf_XYZ = icrs_XYZ @ R.T
        bfsf_XYZ -= np.mean(bfsf_XYZ, axis=0)
        bfsf_XY = bfsf_XYZ[:, :2]
        XY_baseline = linalg.norm(
            bfsf_XY[:, np.newaxis, :] - bfsf_XY[np.newaxis, :, :], axis=-1
        )

        N = sp.jv_series_threshold((2 * np.pi / wl) * XY_baseline.max())
        return 2 * N + 1


class LofarBlock(EarthBoundInstrumentGeometryBlock):
    """
    `LOw-Frequency ARray (LOFAR) <http://www.lofar.org/>`_ located in Europe.
    This LOFAR model consists of 62 stations, each containing between 17 to 24 HBA dipole antennas.

    Args
        N_station : int
            Number of stations to use. (Default = all)

            Sometimes only a subset of an instrument’s stations are desired.
            Setting `N_station` limits the number of stations to those that appear first in `XYZ`
            when sorted by STATION_ID.

        station_only : bool
            If :py:obj:`True`, model LOFAR stations as single-element antennas. (Default = False)
    """

    def __init__(self, N_station=None, station_only=False):
        XYZ = self._get_geometry(N_station, station_only)
        super().__init__(XYZ, N_station)

    def _get_geometry(self, N_station, station_only):
        """
        Load instrument geometry.

        Returns
            :py:class:`~bipp.phased_array.instrument.InstrumentGeometry`
                ITRS instrument geometry.
        """
        rel_path = pathlib.Path("data", "instrument", "LOFAR.csv")
        abs_path = pkg.resource_filename("bipp", str(rel_path))

        itrs_geom = pd.read_csv(abs_path).set_index(["STATION_ID", "ANTENNA_ID"])

        N_station_conf = len(itrs_geom.index.unique(level='STATION_ID'))

        if N_station:
            _check_requested_geometry(N_station_conf, N_station, abs_path)

        if station_only:
            itrs_geom = itrs_geom.groupby("STATION_ID").mean()
            station_id = itrs_geom.index.get_level_values("STATION_ID")
            itrs_geom.index = pd.MultiIndex.from_product(
                [station_id, [0]], names=["STATION_ID", "ANTENNA_ID"]
            )

        XYZ = _as_InstrumentGeometry(itrs_geom)
        return XYZ


class MwaBlock(EarthBoundInstrumentGeometryBlock):
    """
    `Murchison Widefield Array (MWA) <http://www.mwatelescope.org/>`_ located in Australia.

    MWA consists of 128 stations, each containing 16 dipole antennas.

    Args
        N_station : int
            Number of stations to use. (Default = all)

            Sometimes only a subset of an instrument’s stations are desired.
            Setting `N_station` limits the number of stations to those that appear first in `XYZ`
            when sorted by STATION_ID.

        station_only : bool
            If :py:obj:`True`, model MWA stations as single-element antennas. (Default = False)
    """

    def __init__(self, N_station=None, station_only=False):
        XYZ = self._get_geometry(N_station, station_only)
        super().__init__(XYZ, N_station)

    def _get_geometry(self, N_station, station_only):
        """
        Load instrument geometry.

        Args
            station_only : bool
                If :py:obj:`True`, model stations as single-element antennas.

        Returns
            :py:class:`~bipp.phased_array.instrument.InstrumentGeometry`
                ITRS instrument geometry.
        """
        rel_path = pathlib.Path("data", "instrument", "MWA.csv")
        abs_path = pkg.resource_filename("bipp", str(rel_path))

        itrs_geom = pd.read_csv(abs_path).set_index("STATION_ID")

        N_station_conf = len(itrs_geom.index.unique(level='STATION_ID'))

        if N_station:
            _check_requested_geometry(N_station_conf, N_station, abs_path)

        station_id = itrs_geom.index.get_level_values("STATION_ID")
        if station_only:
            itrs_geom.index = pd.MultiIndex.from_product(
                [station_id, [0]], names=["STATION_ID", "ANTENNA_ID"]
            )
        else:
            # Generate flat 4x4 antenna grid pointing towards the Noth pole.
            x_lim = y_lim = 1.65
            lY, lX = np.meshgrid(
                np.linspace(-y_lim, y_lim, 4),
                np.linspace(-x_lim, x_lim, 4),
                indexing="ij",
            )
            l = np.stack((lX, lY, np.zeros((4, 4))), axis=0)

            # For each station: rotate 4x4 array to lie on the sphere's surface.
            xyz_station = itrs_geom.loc[:, ["X", "Y", "Z"]].values
            df_stations = []
            for st_id, st_cog in zip(station_id, xyz_station):
                _, st_colat, st_lon = transform.cart2pol(*st_cog)
                st_cog_unit = transform.pol2cart(1, st_colat, st_lon).reshape(-1)

                R_1 = pylinalg.rot([0, 0, 1], st_lon)
                R_2 = pylinalg.rot(
                    axis=np.cross([0, 0, 1], st_cog_unit), angle=st_colat
                )
                R = R_2 @ R_1

                st_layout = np.reshape(
                    st_cog.reshape(3, 1, 1) + np.tensordot(R, l, axes=1), (3, -1)
                )
                idx = pd.MultiIndex.from_product(
                    [[st_id], range(16)], names=["STATION_ID", "ANTENNA_ID"]
                )
                df_stations += [
                    pd.DataFrame(data=st_layout.T, index=idx, columns=["X", "Y", "Z"])
                ]
            itrs_geom = pd.concat(df_stations)

        XYZ = _as_InstrumentGeometry(itrs_geom)
        return XYZ
