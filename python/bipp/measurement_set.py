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


@chk.check(
    dict(S=chk.is_instance(vis.VisibilityMatrix), W=chk.is_instance(beamforming.BeamWeights))
)
def filter_data(S, W):
    """
    Fix mis-matches to make data streams compatible.

    Visibility matrices from MS files typically include broken beams and/or may not match beams
    specified in beamforming matrices.
    This mis-match causes computations further down the imaging pypeline to be less efficient or
    completely wrong.
    This function applies 2 corrections to visibility and beamforming matrices to make them
    compliant:

    * Drop beams in `S` that do not appear in `W`;
    * Insert 0s in `W` where `S` has broken beams.

    Parameters
    ----------
    S : :py:class:`~pypeline.phased_array.data_gen.statistics.VisibilityMatrix`
        (N_beam1, N_beam1) visibility matrix.
    W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
        (N_antenna, N_beam2) beamforming matrix.

    Returns
    -------
    S : :py:class:`~pypeline.phased_array.data_gen.statistics.VisibilityMatrix`
        (N_beam2, N_beam2) filtered visibility matrix.
    W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
        (N_antenna, N_beam2) filtered beamforming matrix.
    """
    # Stage 1: Drop beams in S that do not appear in W
    beam_idx1 = pd.Index(S.index[0], name="BEAM_ID")
    beam_idx2 = W.index[1]
    beams_to_drop = beam_idx1.difference(beam_idx2)
    beams_to_keep = beam_idx1.drop(beams_to_drop)

    mask = np.any(beam_idx1.values.reshape(-1, 1) == beams_to_keep.values.reshape(1, -1), axis=1)
    S_f = vis.VisibilityMatrix(data=S.data[np.ix_(mask, mask)], beam_idx=beam_idx1[mask])

    # Stage 2: Insert 0s in W where S had broken beams
    broken_beam_idx = beam_idx2[np.isclose(np.sum(S_f.data, axis=1), 0)]
    mask = np.any(beam_idx2.values.reshape(-1, 1) == broken_beam_idx.values.reshape(1, -1), axis=1)

    if np.any(mask) and sparse.isspmatrix(W.data):
        w_lil = W.data.tolil()  # for efficiency
        w_lil[:, mask] = 0
        w_f = w_lil.tocsr()
    else:
        w_f = W.data.copy()
        w_f[:, mask] = 0
    W_f = beamforming.BeamWeights(data=w_f, ant_idx=W.index[0], beam_idx=beam_idx2)

    return S_f, W_f


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
        )
    )
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

        t = ct.table(self._msf, readonly=True)
        time = np.array(t.calc("MJD(TIME)"))
        ant1 = np.array(t.getcol("ANTENNA1"))
        ant2 = np.array(t.getcol("ANTENNA2"))
        dt = np.array(t.getcol(column))
        flag = np.array(t.getcol("FLAG"))

        utime, idx, cnt = np.unique(time, return_index=True, return_counts=True)

        if isinstance(time_id, int):
            time_id = slice(time_id, time_id + 1, 1)
        N_time = len(time)
        time_start, time_stop, time_step = time_id.indices(N_time)

        utime = time[time_start: time_stop: time_step]

        for k in range(len(utime)):
            start=idx[k]
            end=start+cnt[k]
            beam_id_0 = ant1[start:end]
            beam_id_1 = ant2[start:end]
            data_flag = flag[start:end]
            data = dt[start:end]


            # We only want XX and YY correlations
            data = np.average(data[:, :, [0, 3]], axis=2)[:, channel_id]
            data_flag = np.any(data_flag[:, :, [0, 3]], axis=2)[:, channel_id]
            data[data_flag] = 0

            beam_id = np.unique(self.instrument._layout.index.get_level_values("STATION_ID"))
            N_beam = len(beam_id)
            i, j = np.triu_indices(N_beam, k=0)

            row_id_wanted = beam_id[i]
            col_id_wanted = beam_id[j]

            mask = np.logical_and(np.isin(beam_id_0, row_id_wanted), np.isin(beam_id_1, col_id_wanted))


            for ch in channel_id:
                # Apply the mask to retain only the desired pairs
                filtered_col_id_full = beam_id_1[mask]
                filtered_row_id_full = beam_id_0[mask]
                filtered_data = data[:,ch][mask]

                matrix_size = max(np.max(filtered_row_id_full), np.max(filtered_col_id_full)) + 1

                matrix = np.zeros((matrix_size, matrix_size),dtype=np.complex64)

                matrix[filtered_row_id_full, filtered_col_id_full] = filtered_data
                matrix[filtered_col_id_full, filtered_row_id_full] = np.conjugate(filtered_data)

                # Find the row and column indices where the entire row and column are zero
                zero_rows = np.where(~matrix.any(axis=1))[0]
                zero_columns = np.where(~matrix.any(axis=0))[0]
                non_zero_index = np.where(matrix.any(axis=1))[0]

                # Remove the zero rows and columns
                v = np.delete(matrix, zero_rows, axis=0)
                v = np.delete(v, zero_columns, axis=1)
                
                f = self.channels["FREQUENCY"]
                
                vismatrix = vis.VisibilityMatrix(v, non_zero_index)
                
                yield utime[k], f[ch], vismatrix


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


class OskarMeasurementSet(MeasurementSet):
    """
    OSKAR Measurement Set reader.
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
            print("OSKAR XYZ:\n", XYZ)
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
             N_station=chk.allow_None(chk.is_integer),
             origin=chk.allow_None(chk.is_instance(coord.EarthLocation)))
    )
    def __init__(self, file_name, N_station=None, origin=None):
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
        origin : astropy EarthLocation
            Reference location used to compute local station coordinates (ref. RASCIL issue)
        """
        super().__init__(file_name)
        if N_station is not None:
            if N_station <= 0:
                raise ValueError("Parameter[N_station] must be positive.")
        self._N_station = N_station
        self._origin = origin

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
            #print(cfg_idx)
            #print(cfg)
            #import sys
            #sys.exit(1)


            if self._origin:
                o = np.array([self._origin.x.value, self._origin.y.value, self._origin.z.value])
                xyz = cfg.values - o
                for i in range(0, xyz.shape[0]):
                    xyz[i,:] = rascil_crd__enu_to_ecef(self._origin, xyz[i,:])
                XYZ = instrument.InstrumentGeometry(xyz=xyz, ant_idx=cfg.index)
                #XYZ_wrong = instrument.InstrumentGeometry(xyz=cfg.values, ant_idx=cfg.index)
                #print("XYZ CORRECT\n", XYZ.data[0:5,:])
                #print("XYZ WRONG\n", XYZ_wrong.data[0:5,:])
                #print("XYZ DIFF\n", XYZ_wrong.data[0:5,:] - XYZ.data[0:5,:])
            else:
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
