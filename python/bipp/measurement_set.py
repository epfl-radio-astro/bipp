# #############################################################################
# measurement_set.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Measurement Set (MS) readers and tools.
"""

import pathlib
import astropy.coordinates as coord
import astropy.table as tb
import astropy.time as time
import astropy.units as u
import casacore.tables as ct
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import bipp.imot_tools.util.argcheck as chk
import bipp.beamforming as beamforming
import bipp.instrument as instrument
import bipp.statistics as vis
from time import perf_counter as pcf
import sys


def time_idx_in_slice(slice_in, idx):
    if slice_in.stop < -1:
        raise ValueError(f"Slice end value must not be smaller than -1")

    if idx < slice_in.start:
        return False
    if (idx - slice_in.start) % slice_in.step != 0:
        return False
    if slice_in.stop == -1:
        return True
    if idx >= slice_in.stop:
        return False

    return True


def check_continuity(list):
    return not any(a + 1 != b for a, b in zip(list, list[1:]))


# Define chunks of size chunk_size. chunk_size should be optimized for fastest
# block reading (see readms app from casacore for testing around -chansize paramater)
def chunk_channel_block(channel_block, chunk_size):
    block_size = len(channel_block)
    chan0 = channel_block[0]
    for chunk_start in range(chan0, chan0 + block_size, chunk_size):
        chunk_end = chunk_start + chunk_size - 1
        if (chunk_end >= chan0 + block_size):
            chunk_end = chan0 + block_size - 1
        yield [chunk_start, chunk_end]


class MeasurementSet:
    """
    MS file reader.

    This class contains the high-level interface all sub-classes must implement.

    Focus is given to reading MS files from phased-arrays for the moment (i.e, not dish arrays).
    """

    @chk.check("file_name", chk.is_instance(str))
    def __init__(self, file_name):
        """
        Parameters
        ----------
        file_name : str
            Name of the MS file.
        """
        path = pathlib.Path(file_name).absolute()

        if not path.exists():
            raise FileNotFoundError(f"{file_name} does not exist.")

        if not path.is_dir():
            raise NotADirectoryError(f"{file_name} is not a directory, so cannot be an MS file.")

        self._msf = str(path)

        # Buffered attributes
        self._field_center = None
        self._channels = None
        self._time = None
        self._instrument = None
        self._beamformer = None

    @property
    def field_center(self):
        """
        Returns
        -------
        :py:class:`~astropy.coordinates.SkyCoord`
            Observed field's center.
        """

        if self._field_center is None:
            # Following the MS file specification from https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/measurement-set,
            # the FIELD sub-table contains REFERENCE_DIR which (for interferometers) gives the field's direction.
            # It is generally encoded in (lon[rad], lat[rad]) format and given in ICRS coordinates,
            # reason why the TIME field present in the FIELD sub-table is not needed.
            # One must take care to verify the encoding scheme for different MS files as different
            # conventions may be used.
            query = f"select REFERENCE_DIR, TIME from {self._msf}::FIELD"
            table = ct.taql(query)

            lon, lat = table.getcell("REFERENCE_DIR", 0).flatten()
            self._field_center = coord.SkyCoord(ra=lon * u.rad, dec=lat * u.rad, frame="icrs")

        return self._field_center

    @property
    def channels(self):
        """
        Frequency channels available.

        Returns
        -------
        :py:class:`~astropy.table.QTable`
            (N_channel, 2) table with columns

            * CHANNEL_ID : int
            * FREQUENCY : :py:class:`~astropy.units.Quantity`
        """
        if self._channels is None:
            # Following the MS file specification from https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/measurement-set,
            # the SPECTRAL_WINDOW sub-table contains CHAN_FREQ which gives the center frequency for
            # each channel.
            # It is generally encoded in [Hz].
            # One must take care to verify the encoding scheme for different MS files as different
            # conventions may be used.
            query = f"select CHAN_FREQ, CHAN_WIDTH from {self._msf}::SPECTRAL_WINDOW"
            table = ct.taql(query)
            f = table.getcell("CHAN_FREQ", 0).flatten() * u.Hz
            f_id = range(len(f))
            self._channels = tb.QTable(dict(CHANNEL_ID=f_id, FREQUENCY=f))

        return self._channels

    @property
    def time(self):
        """
        Visibility acquisition times.

        Returns
        -------
        :py:class:`~astropy.table.QTable`
            (N_time, 2) table with columns

            * TIME_ID : int
            * TIME : :py:class:`~astropy.time.Time`
        """
        if self._time is None:
            # Following the MS file specification from https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/measurement-set,
            # the MAIN table contains TIME which gives the integration midpoints.
            # It is generally given in UTC scale (in seconds) with epoch set at 0 MJD (Modified Julian Date).
            # Only the TIME column of the MAIN table is required, so we could use the TaQL query below:
            #    select unique MJD(TIME) as MJD_TIME from {self._msf} orderby TIME
            #
            # Unfortunately this query consumes a lot of memory due to the column selection process.
            # Therefore, we will instead ask for all columns and only access the one of interest.
            query = f"select * from {self._msf}"
            table = ct.taql(query)

            t = time.Time(np.unique(table.calc("MJD(TIME)")), format="mjd", scale="utc")
            t_id = range(len(t))
            self._time = tb.QTable(dict(TIME_ID=t_id, TIME=t))

        return self._time

    @property
    def instrument(self):
        """
        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock`
            Instrument position computer.
        """
        raise NotImplementedError

    @property
    def beamformer(self):
        """
        Each dataset has been beamformed in a specific way.
        This property outputs the correct beamformer to compute the beamforming weights.

        Returns
        -------
        :py:class:`~pypeline.phased_array.beamforming.BeamformerBlock`
            Beamweight computer.
        """
        raise NotImplementedError

    @chk.check(
        dict(
            channel_id=chk.accept_any(chk.has_integers, chk.is_instance(slice)),
            time_id=chk.accept_any(chk.is_integer, chk.is_instance(slice)),
            column=chk.is_instance(str),
            sort_time=chk.is_boolean,
            log_level=chk.is_integer
        )
    )
    def visibilities(self, channel_id, time_id, column, sort_time=True, log_level=0):

        query = f"select NAME from {self._msf}::ANTENNA"
        antenna_table = ct.taql(query)
        n_ant = len(antenna_table.getcol("NAME"))

        if column not in ct.taql(f"select * from {self._msf}").colnames():
            raise ValueError(f"column={column} does not exist in {self._msf}::MAIN.")

        # Check whether channels is a single block
        channel_block = True
        if type(channel_id) == slice:
            if (abs(channel_id.step) != 1):
                channel_block = False
        else:
            channel_id = sorted(channel_id)
            nchan = len(channel_id)
            if nchan > 1:
                channel_block = check_continuity(channel_id)

        channel_id = self.channels["CHANNEL_ID"][channel_id]
        if channel_block:
            block_size = len(channel_id)
            if block_size == 0:
                raise Exception("Empty block of channels to process")

        # Create beam index, time independent
        beam_id = np.unique(self.instrument._layout.index.get_level_values("STATION_ID"))
        beam_idx = pd.Index(beam_id, name="BEAM_ID")

        if chk.is_integer(time_id):
            time_id = slice(time_id, time_id + 1, 1)

        # Start reading the table
        table = ct.table(self._msf)

        # Allocate once, size is time invariant, but reset to zero while iterating
        S  = np.zeros((n_ant, n_ant), dtype=complex)
        S.setflags(write=1)
        WS = np.zeros((n_ant, n_ant), dtype=float)

        time_sorted = True
        previous_time = 0
        empty_chan_ep = 0

        for idx, sub_table in enumerate(table.iter("TIME", sort=sort_time)):

            # Skip unwanted epochs
            if not time_idx_in_slice(time_id, idx):
                continue

            t = time.Time(sub_table.calc("MJD(TIME)")[0], format="mjd", scale="utc")
            if t.to_value('mjd') < previous_time:
                time_sorted = False
            previous_time = t.to_value('mjd')

            f = self.channels["FREQUENCY"]

            ant1 = sub_table.getcol("ANTENNA1")  # (N_entry,)
            ant2 = sub_table.getcol("ANTENNA2")  # (N_entry,)

            # If processing a (single) block of channels, then chunk it and loop over chunks
            # to read the data. Otherwise, loop over single channels
            # TODO(EO): check whether blocking would help for close enough channels.
            chunk_size = 8
            if channel_block:
                for chunk in chunk_channel_block(channel_id, chunk_size):
                    data = sub_table.getcolslice(column, blc=(chunk[0], 0), trc=(chunk[1], 3), inc=(1,3))
                    data = np.average(data[:, :, [0,1]], axis=2)
                    flag = sub_table.getcolslice('FLAG', blc=(chunk[0], 0), trc=(chunk[1], 3), inc=(1,3))
                    flag = np.any(flag[:, :, [0,1]], axis=2)
                    data[flag] = 0
                    try:
                        weight_spectrum = sub_table.getcolslice('WEIGHT_SPECTRUM', blc=(chunk[0], 0), trc=(chunk[1], 3), inc=(1,3))
                    except:
                        weight_spectrum = None

                    if weight_spectrum is not None:
                        # Mimic WSClean
                        weight = np.min(weight_spectrum[:, :, [0,1]], axis=2)
                        weight[flag] = 0

                    for i in range(0, (chunk[1] - chunk[0] + 1)):
                        S.fill(0)
                        S[ant2, ant1] = data[:, i]
                        if weight_spectrum is not None and np.any(weight[:, i]):
                            WS.fill(0)
                            WS[ant2, ant1] = weight[:, i]
                            vis_mat = vis.VisibilityMatrix(S, beam_idx, check_hermitian=False, weight_spectrum=WS)
                        else:
                            vis_mat = vis.VisibilityMatrix(S, beam_idx, check_hermitian=False)

                        # Skip empty channels
                        if np.count_nonzero(vis_mat.data) == 0:
                            empty_chan_ep += 1
                            continue

                        yield t, f[chunk[0] + i], vis_mat
            else:
                for chan in channel_id:
                    data = sub_table.getcolslice(column, blc=(chan, 0), trc=(chan, 3), inc=(1,3))
                    data = np.average(data[:, :, [0,1]], axis=2)
                    flag = sub_table.getcolslice('FLAG', blc=(chan, 0), trc=(chan, 3), inc=(1,3))
                    flag = np.any(flag[:, :, [0,1]], axis=2)
                    data[flag] = 0
                    try:
                        weight_spectrum = sub_table.getcolslice('WEIGHT_SPECTRUM', blc=(chan, 0), trc=(chan, 3), inc=(1,3))
                    except:
                        weight_spectrum = None

                    if weight_spectrum is not None:
                        # Mimic WSClean
                        weight = np.min(weight_spectrum[:, :, [0,1]], axis=2)
                        weight[flag] = 0

                    # Skip empty channels
                    if np.count_nonzero(data[:, 0]) == 0:
                        continue

                    S.fill(0)
                    S[ant2, ant1] = data[:, 0]

                    if weight_spectrum is not None and np.any(weight):
                        WS.fill(0)
                        WS[ant2, ant1] = weight[:, 0]
                        vis_mat = vis.VisibilityMatrix(S, beam_idx, check_hermitian=False, weight_spectrum=WS)
                    else:
                        vis_mat = vis.VisibilityMatrix(S, beam_idx, check_hermitian=False)

                    yield t, f[chan], vis_mat

        if log_level > 0:
            print(f"-I- Was sorting time requested? {sort_time}; was TIME sorted? {time_sorted}. Empty channels at times =", empty_chan_ep)


def _series2array(visibility: pd.Series) -> np.ndarray:
    b_idx_0 = visibility.index.get_level_values("B_0").to_series()
    b_idx_1 = visibility.index.get_level_values("B_1").to_series()

    row_map = (
        pd.concat(objs=(b_idx_0, b_idx_1), ignore_index=True)
        .drop_duplicates()
        .to_frame(name="BEAM_ID")
        .assign(ROW_ID=lambda df: np.arange(len(df)))
    )
    col_map = row_map.rename(columns={"ROW_ID": "COL_ID"})

    data = (
        visibility.reset_index()
        .merge(row_map, left_on="B_0", right_on="BEAM_ID")
        .merge(col_map, left_on="B_1", right_on="BEAM_ID")
        .loc[:, ["ROW_ID", "COL_ID", "S"]]
    )

    N_beam = len(row_map)
    S = np.zeros(shape=(N_beam, N_beam), dtype=complex)
    S[data.ROW_ID.values, data.COL_ID.values] = data.S.values
    S_diag = np.diag(S)
    S = S + S.conj().T
    S[np.diag_indices_from(S)] = S_diag
    return S

def _series2array_w(weight: pd.Series) -> np.ndarray:
    b_idx_0 = weight.index.get_level_values("B_0").to_series()
    b_idx_1 = weight.index.get_level_values("B_1").to_series()

    row_map = (
        pd.concat(objs=(b_idx_0, b_idx_1), ignore_index=True)
        .drop_duplicates()
        .to_frame(name="BEAM_ID")
        .assign(ROW_ID=lambda df: np.arange(len(df)))
    )
    col_map = row_map.rename(columns={"ROW_ID": "COL_ID"})

    data = (
        weight.reset_index()
        .merge(row_map, left_on="B_0", right_on="BEAM_ID")
        .merge(col_map, left_on="B_1", right_on="BEAM_ID")
        .loc[:, ["ROW_ID", "COL_ID", "W"]]
    )

    N_beam = len(row_map)
    W = np.zeros(shape=(N_beam, N_beam), dtype=float)
    W[data.ROW_ID.values, data.COL_ID.values] = data.W.values
    W_diag = np.diag(W)
    W = W + W.T
    W[np.diag_indices_from(W)] = W_diag
    return W


class LofarMeasurementSet(MeasurementSet):
    """
    LOw-Frequency ARray (LOFAR) Measurement Set reader.
    """

    @chk.check(
        dict(
            file_name=chk.is_instance(str),
            N_station=chk.allow_None(chk.is_integer),
            station_only=chk.is_boolean,
        )
    )
    def __init__(self, file_name, N_station=None, station_only=False):
        """
        Parameters
        ----------
        file_name : str
            Name of the MS file.
        N_station : int
            Number of stations to use. (Default = all)

            Sometimes only a subset of an instrument’s stations are desired.
            Setting `N_station` limits the number of stations to those that appear first when sorted
            by STATION_ID.
        station_only : bool
            If :py:obj:`True`, model LOFAR stations as single-element antennas. (Default = False)
        """
        super().__init__(file_name)

        if N_station is not None:
            if N_station <= 0:
                raise ValueError("Parameter[N_station] must be positive.")
        self._N_station = N_station
        self._station_only = station_only

    @property
    def instrument(self):
        """
        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock`
            Instrument position computer.
        """
        if self._instrument is None:
            # Following the LOFAR MS file specification from https://www.astron.nl/lofarwiki/lib/exe/fetch.php?media=public:documents:ms2_description_for_lofar_2.08.00.pdf,
            # the special LOFAR_ANTENNA_FIELD sub-table must be used due to the hierarchical design
            # of LOFAR.
            # Some remarks on the required fields:
            # - ANTENNA_ID: equivalent to STATION_ID field in `InstrumentGeometry.index[0]`.
            # - POSITION: absolute station positions in ITRF coordinates.
            #             This does not necessarily correspond to the station centroid.
            # - ELEMENT_OFFSET: offset of each antenna in a station.
            #                   When combined with POSITION, it gives the absolute antenna positions
            #                   in ITRF.
            # - ELEMENT_FLAG: True/False value for each (station, antenna, polarization) pair.
            #                 If any of the polarization flags is True for a given antenna, then the
            #                 antenna can be discarded from that station.
            query = f"select ANTENNA_ID, POSITION, ELEMENT_OFFSET, ELEMENT_FLAG from {self._msf}::LOFAR_ANTENNA_FIELD"
            table = ct.taql(query)

            station_id = table.getcol("ANTENNA_ID")
            station_mean = table.getcol("POSITION")
            antenna_offset = table.getcol("ELEMENT_OFFSET")
            antenna_flag = table.getcol("ELEMENT_FLAG")

            # Form DataFrame that holds all antennas, then filter out flagged antennas.
            N_station, N_antenna, _ = antenna_offset.shape
            station_mean = np.reshape(station_mean, (N_station, 1, 3))
            antenna_xyz = np.reshape(station_mean + antenna_offset, (N_station * N_antenna, 3))
            antenna_flag = np.reshape(antenna_flag.any(axis=2), (N_station * N_antenna))

            cfg_idx = pd.MultiIndex.from_product(
                [station_id, range(N_antenna)], names=("STATION_ID", "ANTENNA_ID")
            )
            cfg = pd.DataFrame(data=antenna_xyz, columns=("X", "Y", "Z"), index=cfg_idx).loc[
                ~antenna_flag
            ]

            # If in `station_only` mode, return centroid of each station only.
            # Why do we not just use `station_mean` above? Because it arbitrarily
            # points to some sub-antenna, not the station centroid.
            if self._station_only:
                cfg = cfg.groupby("STATION_ID").mean()
                station_id = cfg.index.get_level_values("STATION_ID")
                cfg.index = pd.MultiIndex.from_product(
                    [station_id, [0]], names=["STATION_ID", "ANTENNA_ID"]
                )

            # Finally, only keep the stations that were specified in `__init__()`.
            XYZ = instrument.InstrumentGeometry(xyz=cfg.values, ant_idx=cfg.index)
            self._instrument = instrument.EarthBoundInstrumentGeometryBlock(XYZ, self._N_station)

        return self._instrument

    @property
    def beamformer(self):
        """
        Each dataset has been beamformed in a specific way.
        This property outputs the correct beamformer to compute the beamforming weights.

        Returns
        -------
        :py:class:`~pypeline.phased_array.beamforming.MatchedBeamformerBlock`
            Beamweight computer.
        """
        if self._beamformer is None:
            # LOFAR uses Matched-Beamforming exclusively, with a single beam output per station.
            XYZ = self.instrument._layout
            beam_id = np.unique(XYZ.index.get_level_values("STATION_ID"))

            direction = self.field_center
            beam_config = [(_, _, direction) for _ in beam_id]
            self._beamformer = beamforming.MatchedBeamformerBlock(beam_config)

        return self._beamformer


class MwaMeasurementSet(MeasurementSet):
    """
    Murchison Widefield Array (MWA) Measurement Set reader.
    """

    @chk.check("file_name", chk.is_instance(str))
    def __init__(self, file_name):
        """
        Parameters
        ----------
        file_name : str
            Name of the MS file.
        """
        super().__init__(file_name)

    @property
    def instrument(self):
        """
        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock`
            Instrument position computer.
        """
        if self._instrument is None:
            # Following the MS file specification from https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/measurement-set,
            # the ANTENNA sub-table specifies the antenna geometry.
            # Some remarks on the required fields:
            # - POSITION: absolute station positions in ITRF coordinates.
            # - ANTENNA_ID: equivalent to STATION_ID field `InstrumentGeometry.index[0]`
            #               This field is NOT present in the ANTENNA sub-table, but is given
            #               implicitly by its row-ordering.
            #               In other words, the station corresponding to ANTENNA1=k in the MAIN
            #               table is described by the k-th row of the ANTENNA sub-table.
            query = f"select POSITION from {self._msf}::ANTENNA"
            table = ct.taql(query)
            station_mean = table.getcol("POSITION")

            N_station = len(station_mean)
            station_id = np.arange(N_station)
            cfg_idx = pd.MultiIndex.from_product(
                [station_id, [0]], names=("STATION_ID", "ANTENNA_ID")
            )
            cfg = pd.DataFrame(data=station_mean, columns=("X", "Y", "Z"), index=cfg_idx)

            XYZ = instrument.InstrumentGeometry(xyz=cfg.values, ant_idx=cfg.index)

            self._instrument = instrument.EarthBoundInstrumentGeometryBlock(XYZ)

        return self._instrument

    @property
    def beamformer(self):
        """
        Each dataset has been beamformed in a specific way.
        This property outputs the correct beamformer to compute the beamforming weights.

        Returns
        -------
        :py:class:`~pypeline.phased_array.beamforming.MatchedBeamformerBlock`
            Beamweight computer.
        """
        if self._beamformer is None:
            # MWA does not do any beamforming.
            # Given the single-antenna station model in MS files from MWA, this can be seen as
            # Matched-Beamforming, with a single beam output per station.
            XYZ = self.instrument._layout
            beam_id = np.unique(XYZ.index.get_level_values("STATION_ID"))

            direction = self.field_center
            beam_config = [(_, _, direction) for _ in beam_id]
            self._beamformer = beamforming.MatchedBeamformerBlock(beam_config)

        return self._beamformer


class SKALowMeasurementSet(MeasurementSet):
    """
    SKA Low Measurement Set reader.
    """
    @chk.check(
        dict(file_name=chk.is_instance(str),
             N_station=chk.allow_None(chk.is_integer))
    )
    def __init__(self, file_name, N_station=None):
        """
        Parameters
        ----------
        file_name : str
            Name of the MS file.
        N_station : int
            Number of stations to use. (Default = all)

            Sometimes only a subset of an instrument’s stations are desired.
            Setting `N_station` limits the number of stations to those that appear first when sorted
            by STATION_ID.
        """
        super().__init__(file_name)
        if N_station is not None:
            if N_station <= 0:
                raise ValueError("Parameter[N_station] must be positive.")
        self._N_station = N_station

    @property
    def instrument(self):
        """
        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock`
            Instrument position computer.
        """
        
        if self._instrument is None:
            # Following the MS file specification from https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/measurement-set,
            # the ANTENNA sub-table specifies the antenna geometry.
            # Some remarks on the required fields:
            # - POSITION: absolute station positions in ITRF coordinates.
            # - ANTENNA_ID: equivalent to STATION_ID field `InstrumentGeometry.index[0]`
            #               This field is NOT present in the ANTENNA sub-table, but is given
            #               implicitly by its row-ordering.
            #               In other words, the station corresponding to ANTENNA1=k in the MAIN
            #               table is described by the k-th row of the ANTENNA sub-table.
            query = f"select POSITION from {self._msf}::ANTENNA"
            table = ct.taql(query)
            station_mean = table.getcol("POSITION")

            N_station = len(station_mean)
            station_id = np.arange(N_station)
            cfg_idx = pd.MultiIndex.from_product(
                [station_id, [0]], names=("STATION_ID", "ANTENNA_ID")
            )
            cfg = pd.DataFrame(data=station_mean, columns=("X", "Y", "Z"), index=cfg_idx)

            XYZ = instrument.InstrumentGeometry(xyz=cfg.values, ant_idx=cfg.index)
            self._instrument = instrument.EarthBoundInstrumentGeometryBlock(XYZ)

        return self._instrument
    
    @property
    def beamformer(self):
        """
        Each dataset has been beamformed in a specific way.
        This property outputs the correct beamformer to compute the beamforming weights.
        Returns
        -------
        :py:class:`~pypeline.phased_array.beamforming.MatchedBeamformerBlock`
            Beamweight computer.
        """
        if self._beamformer is None:
            XYZ = self.instrument._layout
            beam_id = np.unique(XYZ.index.get_level_values("STATION_ID"))
            direction = self.field_center
            beam_config = [(_, _, direction) for _ in beam_id]
            self._beamformer = beamforming.MatchedBeamformerBlock(beam_config)

        return self._beamformer
    
    @property
    def beamformer_identity(self):
        """
        Each dataset has been beamformed in a specific way.
        This property outputs the correct beamformer to compute the beamforming weights.
        Returns
        -------
        :py:class:`~pypeline.phased_array.beamforming.MatchedBeamformerBlock`
            Beamweight computer.
        """
        if self._beamformer is None:
            XYZ = self.instrument._layout
            beam_id = np.unique(XYZ.index.get_level_values("STATION_ID"))
            direction = self.field_center
            beam_config = [(_, _, direction) for _ in beam_id]
            self._beamformer = beamforming.MatchedBeamformerBlockIdentity(beam_config)

        return self._beamformer
