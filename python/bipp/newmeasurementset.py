import pathlib

import astropy.coordinates as coord
import astropy.table as tb
import astropy.time as time
import astropy.units as u
import casacore.tables as ct
import imot_tools.util.argcheck as chk
import numpy as np
import pandas as pd 
from scipy.sparse import coo_matrix
from casacore import tables
from bipp import statistics as vis


class MeasurementSet:
    """
    MS file reader.

    This class contains the high-level interface all sub-classes must implement.

    Focus is given to reading MS files from phased-arrays for the moment (i.e, not dish arrays).
    """

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


    def visibilities(self, channel_id, time_id, column):
        """
        Extract visibility matrices.

        Parameters
        ----------
        channel_id : array-like(int) or slice
            Several CHANNEL_IDs from :py:attr:`~pypeline.phased_array.util.measurement_set.MeasurementSet.channels`.
        time_id : int or slice
            Several TIME_IDs from :py:attr:`~pypeline.phased_array.util.measurement_set.MeasurementSet.time`.
        column : str
            Column name from MAIN table where visibility data resides.

            (This is required since several visibility-holding columns can co-exist.)

        Returns
        -------
        iterable

            Generator object returning (time, freq, S) triplets with:

            * time (:py:class:`~astropy.time.Time`): moment the visibility was formed;
            * freq (:py:class:`~astropy.units.Quantity`): center frequency of the visibility;
            * S (:py:class:`~pypeline.phased_array.data_gen.statistics.VisibilityMatrix`)
        """
        if column not in ct.taql(f"select * from {self._msf}").colnames():
            raise ValueError(f"column={column} does not exist in {self._msf}::MAIN.")

        channel_id = self.channels["CHANNEL_ID"][channel_id]

        if chk.is_integer(time_id):
            time_id = slice(time_id, time_id + 1, 1)
        N_time = len(self.time)
        time_start, time_stop, time_step = time_id.indices(N_time)

        # Only a subset of the MAIN table's columns are needed to extract visibility information.
        # As such, it makes sense to construct a TaQL query that only extracts the columns of
        # interest as shown below:
        #    select ANTENNA1, ANTENNA2, MJD(TIME) as TIME, {column}, FLAG from {self._msf} where TIME in
        #    (select unique TIME from {self._msf} limit {time_start}:{time_stop}:{time_step})
        # Unfortunately this query consumes a lot of memory due to the column selection process.
        # Therefore, we will instead ask for all columns and only access those of interest.
        query = (
            f"select * from {self._msf} where TIME in "
            f"(select unique TIME from {self._msf} limit {time_start}:{time_stop}:{time_step})"
        )
        table = ct.taql(query)

        for sub_table in table.iter("TIME", sort=True):
            
            beam_id_0 = sub_table.getcol("ANTENNA1")  # (N_entry,)
            beam_id_1 = sub_table.getcol("ANTENNA2")  # (N_entry,)
            data_flag = sub_table.getcol("FLAG")  # (N_entry, N_channel, 4)
            data = sub_table.getcol(column)  # (N_entry, N_channel, 4)

            # We only want XX and YY correlations
            data = np.average(data[:, :, [0, 3]], axis=2)[:, channel_id]
            data_flag = np.any(data_flag[:, :, [0, 3]], axis=2)[:, channel_id]
            data[data_flag] = 0

            # DataFrame description of visibility data.
            # Each column represents a different channel.
            S_full_idx = pd.MultiIndex.from_arrays((beam_id_0, beam_id_1), names=("B_0", "B_1"))
            S_full = pd.DataFrame(data=data, columns=channel_id, index=S_full_idx)


            # Drop rows of `S_full` corresponding to unwanted beams.
            beam_id = np.unique(self.instrument._layout.index.get_level_values("STATION_ID"))
            N_beam = len(beam_id)
            i, j = np.triu_indices(N_beam, k=0)
            wanted_index = pd.MultiIndex.from_arrays((beam_id[i], beam_id[j]), names=("B_0", "B_1"))
            index_to_drop = S_full_idx.difference(wanted_index)
            S_trunc = S_full.drop(index=index_to_drop)
            S = S_trunc.reset_index()
            for ch_id in channel_id:
                matrix_data = S[ch_id]

                matrix_size = max(np.max(S.B_0), np.max(S.B_1)) + 1

                matrix = coo_matrix((matrix_data, (S.B_0, S.B_1)), shape=(matrix_size, matrix_size)).toarray()

                #remove broken_rows
                broken_row_id = np.where(~v.any(axis=1))[0]
                
                v = np.delete(matrix, broken_row_id, axis=0)

                t = time.Time(sub_table.calc("MJD(TIME)")[0], format="mjd", scale="utc")
                f = self.channels["FREQUENCY"][ch_id]

                #visibility = vis.VisibilityMatrix(v, matrix_size)

                yield t, f, S_trunc.index, v

    def visibilities_old(self, channel_id, time_id, column):
        """
        Extract visibility matrices.

        Parameters
        ----------
        channel_id : array-like(int) or slice
            Several CHANNEL_IDs from :py:attr:`~pypeline.phased_array.util.measurement_set.MeasurementSet.channels`.
        time_id : int or slice
            Several TIME_IDs from :py:attr:`~pypeline.phased_array.util.measurement_set.MeasurementSet.time`.
        column : str
            Column name from MAIN table where visibility data resides.

            (This is required since several visibility-holding columns can co-exist.)

        Returns
        -------
        iterable

            Generator object returning (time, freq, S) triplets with:

            * time (:py:class:`~astropy.time.Time`): moment the visibility was formed;
            * freq (:py:class:`~astropy.units.Quantity`): center frequency of the visibility;
            * S (:py:class:`~pypeline.phased_array.data_gen.statistics.VisibilityMatrix`)
        """
        if column not in ct.taql(f"select * from {self._msf}").colnames():
            raise ValueError(f"column={column} does not exist in {self._msf}::MAIN.")

        channel_id = self.channels["CHANNEL_ID"][channel_id]

        if chk.is_integer(time_id):
            time_id = slice(time_id, time_id + 1, 1)
        N_time = len(self.time)
        time_start, time_stop, time_step = time_id.indices(N_time)

        # Only a subset of the MAIN table's columns are needed to extract visibility information.
        # As such, it makes sense to construct a TaQL query that only extracts the columns of
        # interest as shown below:
        #    select ANTENNA1, ANTENNA2, MJD(TIME) as TIME, {column}, FLAG from {self._msf} where TIME in
        #    (select unique TIME from {self._msf} limit {time_start}:{time_stop}:{time_step})
        # Unfortunately this query consumes a lot of memory due to the column selection process.
        # Therefore, we will instead ask for all columns and only access those of interest.
        query = (
            f"select * from {self._msf} where TIME in "
            f"(select unique TIME from {self._msf} limit {time_start}:{time_stop}:{time_step})"
        )
        table = ct.taql(query)

        for sub_table in table.iter("TIME", sort=True):
            
            beam_id_0 = sub_table.getcol("ANTENNA1")  # (N_entry,)
            beam_id_1 = sub_table.getcol("ANTENNA2")  # (N_entry,)
            data_flag = sub_table.getcol("FLAG")  # (N_entry, N_channel, 4)
            data = sub_table.getcol(column)  # (N_entry, N_channel, 4)

            # We only want XX and YY correlations
            data = np.average(data[:, :, [0, 3]], axis=2)[:, channel_id]
            data_flag = np.any(data_flag[:, :, [0, 3]], axis=2)[:, channel_id]
            data[data_flag] = 0

            # DataFrame description of visibility data.
            # Each column represents a different channel.
            S_full_idx = pd.MultiIndex.from_arrays((beam_id_0, beam_id_1), names=("B_0", "B_1"))
            S_full = pd.DataFrame(data=data, columns=channel_id, index=S_full_idx)


            # Drop rows of `S_full` corresponding to unwanted beams.
            beam_id = np.unique(self.instrument._layout.index.get_level_values("STATION_ID"))
            N_beam = len(beam_id)
            i, j = np.triu_indices(N_beam, k=0)
            wanted_index = pd.MultiIndex.from_arrays((beam_id[i], beam_id[j]), names=("B_0", "B_1"))
            index_to_drop = S_full_idx.difference(wanted_index)
            S = S_full.drop(index=index_to_drop)

            # Break S into columns and stream out
            t = time.Time(sub_table.calc("MJD(TIME)")[0], format="mjd", scale="utc")
            f = self.channels["FREQUENCY"]
            beam_idx = pd.Index(beam_id, name="BEAM_ID")
            for ch_id in channel_id:
                matrix = _series2array(S[ch_id].rename("S", inplace=True))
                
                #remove broken_rows
                broken_row_id = np.where(~v.any(axis=1))[0]
                
                v = np.delete(matrix, broken_row_id, axis=0)
                
                #visibility = vis.VisibilityMatrix(v, beam_idx)
                yield t, f[ch_id], beam_idx, visibility

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


