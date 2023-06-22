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

            # Set broken visibilities to 0
            data[data_flag] = 0
            
            for ch_id in channel_id:
                matrix_data = data[:, ch_id]

                matrix_size = max(np.max(beam_id_0), np.max(beam_id_1)) + 1

                v = coo_matrix((matrix_data, (beam_id_0, beam_id_1)), shape=(matrix_size, matrix_size)).toarray()

                t = time.Time(sub_table.calc("MJD(TIME)")[0], format="mjd", scale="utc")
                f = self.channels["FREQUENCY"][ch_id]
            
                visibility = vis.VisibilityMatrix(v, matrix_size)
                
                yield t, f visibility

