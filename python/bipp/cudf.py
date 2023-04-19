# #############################################################################
# mscudf.py
# ==================
# Author : Arpan Das [arpan.das@epfl.ch]
# #############################################################################

"""
Cudf file readers and tools.
"""
import pathlib
import dask
import dask.array as da
import dask_cudf
import cudf
import numpy as np
import cupy as cp
import astropy.coordinates as coord
import astropy.table as tb
import astropy.time as time
import astropy.units as u
import imot_tools.util.argcheck as chk
import pandas as pd
import dask.dataframe as dd

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.data_gen.statistics as vis
import pypeline.phased_array.data_gen.statisticsgpu as visgpu

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
    beam_idx1 = S.index[0]
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


def load_dataframe(allcol = True, use_cols = []):

    """
    Load the dataframe into GPU and returns

    Parameters
    ----------
    allcol : bool
        If allcol = True the whole dataframe will be loaded. Else it will load partial dataframe with columns
        passed as use_cols
    use_cols : list
        list of the columns which should be loaded instead of the whole dataframe 

    Returns
    -------
    cudf dataframe (parquet format)
        Cudf dataframe with the columns wanted
    """
    if allcol == True:
        df = cudf.read_parquet(self._cudf)
    else: 
        df = cudf.read_parquet(self._cudf, usecols = use_cols)

    return df

class Cudfparquet:
    """
    Cudf parquet data format reader.

    This class contains the high-level interface all sub-classes must implement.

    Focus is given to reading parquet files from phased-arrays for the moment (i.e, not dish arrays).
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

        # if not path.is_dir():
        #     raise NotADirectoryError(f"{file_name} is not a directory, so cannot be an parquet file.")

        self._cudf = str(path)
        self._dataframe = dask_cudf.read_parquet(self._cudf, cols=['TIME', 'ANTENNA1', 'ANTENNA2', 'FLAG','DATA'], chunksize = '500 MiB')
        self._field_df = dask_cudf.read_parquet(f"{path.parent}/{path.stem}_{'FIELD'}{path.suffix}")
        self._spectral_window_df = dask_cudf.read_parquet(f"{path.parent}/{path.stem}_{'SPECTRAL_WINDOW'}{path.suffix}")

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
            #path = pathlib.Path(self._cudf).absolute()
            #df = cudf.read_parquet(f"{path.parent}/{path.stem}_{'FIELD'}{path.suffix}")
            df = self._field_df
            lon, lat = df['REFERENCE_DIR'].compute().explode()[0]
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
            #path = pathlib.Path(self._cudf).absolute()
            #df = cudf.read_parquet(f"{path.parent}/{path.stem}_{'SPECTRAL_WINDOW'}{path.suffix}")

            df = self._spectral_window_df
            
            f = df['CHAN_FREQ'].compute().list.leaves.to_numpy() * u.Hz
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
            
            self._time = self._dataframe['TIME'].compute().explode().to_cupy()
            # t = time.Time(np.unique(time_array), format="mjd", scale="utc")
            # t_id = range(len(t))
            # self._time = tb.QTable(dict(TIME_ID=t_id, TIME=t))
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
        df = self._dataframe
        
        if column not in df.columns:
            raise ValueError(f"column={column} does not exist in {self._cudf}::MAIN.")
            
        channel_id = self.channels["CHANNEL_ID"][channel_id]
        
        if chk.is_integer(time_id):
            time_id = slice(time_id, time_id + 1, 1)
        N_time = len(self.time)
        time_start, time_stop, time_step = time_id.indices(N_time)
        
        
        obstime = df['TIME'].compute().explode().to_numpy()
        unique_time = np.unique(obstime)[time_start:time_stop:time_step]
        df2 = df.loc[df['TIME'].isin(unique_time)]

        for t in unique_time:
            df_sub = df2.loc[df2['TIME'] == t].reset_index()
            
            beam_id_0_cp = df_sub.ANTENNA1.compute().to_cupy()
            beam_id_1_cp = df_sub.ANTENNA2.compute().to_cupy()
            data_flag_cp = df_sub.FLAG.compute()
            data_flag_cp = data_flag_cp .list.leaves.to_cupy().reshape(len(data_flag_cp),len(data_flag_cp[0]),len(data_flag_cp[0][0]))
            data_cp = df_sub[column].compute()
            data_cp = data_cp.list.leaves.to_cupy(dtype=cp.float64).reshape(len(data_cp),len(data_cp[0]),len(data_cp[0][0])).view(cp.complex128)
            
            # beam_id_0 = df_sub.ANTENNA1.to_numpy()
            # beam_id_1 = df_sub.ANTENNA2.to_numpy()
            # data_flag = df_sub.FLAG.list.leaves.to_numpy().reshape(len(df_sub.FLAG),len(df_sub.FLAG[0]),len(df_sub.FLAG[0][0]))
            # data = df_sub[column].list.leaves.to_numpy(dtype=np.float32).reshape(len(df_sub.DATA),len(df_sub.DATA[0]),len(df_sub.DATA[0][0])).view(np.complex64)
            
            # We only want XX and YY correlations
            # data = np.average(data[:, :, [0, 3]], axis=2)[:, channel_id]
            # data_flag = np.any(data_flag[:, :, [0, 3]], axis=2)[:, channel_id]
            
            data_cp = cp.average(data_cp[:, :, [0, 3]], axis=2)[:, channel_id]
            data_flag_cp = cp.any(data_flag_cp[:, :, [0, 3]], axis=2)[:, channel_id]
            
            
            # DataFrame description of visibility data.
            # Each column represents a different channel.
#             S_full_idx = pd.MultiIndex.from_arrays((beam_id_0, beam_id_1), names=("B_0", "B_1"))
            
#             S_full = pd.DataFrame(data=data, columns=channel_id, index=S_full_idx)
            
            S_full_cudf = cudf.DataFrame({'B_0': beam_id_0_cp, 'B_1': beam_id_1_cp})
            
            for i in np.array(channel_id):
                S_full_cudf[i] = data_cp.T[i].view(cp.float32).reshape(data_cp.T[i].shape + (2,)).tolist()
                
            
            # Drop rows of `S_full` corresponding to unwanted beams.
            # beam_id = np.unique(self.instrument._layout.index.get_level_values("STATION_ID"))
            # N_beam = len(beam_id)
            # i, j = np.triu_indices(N_beam, k=0)
            
#             wanted_index = pd.MultiIndex.from_arrays((beam_id[i], beam_id[j]), names=("B_0", "B_1"))

#             index_to_drop = S_full_idx.difference(wanted_index)

#             S_trunc = S_full.drop(index=index_to_drop)
            
            beam_id_cp = cp.unique(self.instrument._layout.index.get_level_values("STATION_ID"))
            N_beam_cp = len(beam_id_cp)
            i_cp, j_cp = cp.triu_indices(N_beam_cp, k=0)
            S_trunc_cudf = S_full_cudf[S_full_cudf['B_0'].isin(beam_id_cp[i_cp]) & S_full_cudf['B_1'].isin(beam_id_cp[j_cp])]
            

            # Depending on the dataset, some (ANTENNA1, ANTENNA2) pairs that have correlation=0 are
            # omitted in the table.
            # This is problematic as the previous DataFrame construction could be potentially
            # missing entire antenna ranges.
            # To fix this issue, we augment the dataframe to always make sure `S_trunc` matches the
            # desired shape.
            
            # index_diff = wanted_index.difference(S_trunc.index)
            # N_diff = len(index_diff)

#             S_fill_in = pd.DataFrame(
#                 data=np.zeros((N_diff, len(channel_id)), dtype=data.dtype),
#                 columns=channel_id,
#                 index=index_diff,
#             )
#             S = pd.concat([S_trunc, S_fill_in], axis=0, ignore_index=False).sort_index(
#                 level=["B_0", "B_1"]
#             )
            

            wanted_df = cudf.DataFrame()
            wanted_df['B_0'] = beam_id_cp[i_cp]
            wanted_df['B_1'] = beam_id_cp[j_cp]
            dummy_complexarray = cp.zeros(len(beam_id_cp[i_cp]),dtype=cp.complex64)
            for i in np.array(channel_id):
                wanted_df[i] = dummy_complexarray.view(cp.float32).reshape(dummy_complexarray.T.shape + (2,)).tolist()
                
            S_cudf = cudf.concat([S_trunc_cudf, wanted_df]).drop_duplicates(subset=['B_0','B_1'])
            S_cudf = S_cudf.sort_values(['B_0', 'B_1'], ascending=[True, True])
            S_cudf = S_cudf.reset_index().set_index(['B_0','B_1'])
            
            # Break S into columns and stream out
            #t = df_sub.TIME[0]
            t = time.Time(df_sub.TIME.compute()[0], format="mjd", scale="utc")
            f = self.channels["FREQUENCY"]
            beam_idx = pd.Index(beam_id_cp.get(), name="BEAM_ID")
            #beam_idx = cudf.Index(beam_id_cp, name="BEAM_ID")
            for ch_id in channel_id:
                v_cudf = _series2array_cudf(S_cudf[ch_id].rename("S"))
                #visibility = vis.VisibilityMatrix(v_cudf.get(), beam_idx)
                visibility = visgpu.VisibilityMatrix(v_cudf, beam_idx)
                yield t, f[ch_id], visibility
                

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

def _series2array_cudf(visibility) -> cp.ndarray:
    b_idx_0 = visibility.index.get_level_values("B_0").to_series()
    b_idx_1 = visibility.index.get_level_values("B_1").to_series()
    # b_idx_0 = cudf.Series(visibility.index.get_level_values("B_0"))
    # b_idx_1 = cudf.Series(visibility.index.get_level_values("B_1"))
    
    row_map = (
        cudf.concat(objs=(b_idx_0, b_idx_1), ignore_index=True)
        .drop_duplicates()
        .to_frame(name="BEAM_ID")
    )
    row_map["ROW_ID"] = cp.arange(len(row_map.BEAM_ID))
    
    col_map = row_map.rename(columns={"ROW_ID": "COL_ID"})
    
    data = (
        visibility.reset_index()
        .merge(row_map, left_on="B_0", right_on="BEAM_ID")
        .merge(col_map, left_on="B_1", right_on="BEAM_ID")
        .loc[:, ["ROW_ID", "COL_ID", "S"]]
    )
    N_beam = len(row_map)
    S = cp.zeros(shape=(N_beam, N_beam), dtype=complex)
    S[data.ROW_ID.values, data.COL_ID.values] = data.S.list.leaves.to_cupy(cp.float32).reshape(2*len(data.S)).view(cp.complex64)
    S_diag = cp.diag(S)
    S = S + S.conj().T
    S[cp.diag_indices_from(S)] = S_diag
    return S

    
    


class LofarMeasurementSet(Cudfparquet):
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
        
        path = pathlib.Path(file_name).absolute()
        self._lofar_antenna_field_df = cudf.read_parquet(f"{path.parent}/{path.stem}_{'LOFAR_ANTENNA_FIELD'}{path.suffix}")
        
    
    @property
    def instrument(self):
        """
        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock`
            Instrument position computer.
        """
        if self._instrument is None:
            # Following the LOFAR MS file specification from https://www.astron.nl/lofarwiki/lib/exe/fetch.php? media=public:documents:ms2_description_for_lofar_2.08.00.pdf,
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
            
            
            path = pathlib.Path(self._cudf).absolute()
            df = self._lofar_antenna_field_df
            
            station_id = df.ANTENNA_ID.to_numpy(dtype=np.int32)
            station_mean = df.POSITION.list.leaves.to_numpy().reshape(len(df.POSITION),3)
            antenna_offset = df.ELEMENT_OFFSET.list.leaves.to_numpy().reshape(len(df.ELEMENT_OFFSET),len(df.ELEMENT_OFFSET[0]),len(df.ELEMENT_OFFSET[0][0]))
            antenna_flag = df.ELEMENT_FLAG.list.leaves.to_numpy().reshape(len(df.ELEMENT_FLAG),len(df.ELEMENT_FLAG[0]),len(df.ELEMENT_FLAG[0][0]))
            
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
    
    
class MwaMeasurementSet(Cudfparquet):
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
        path = pathlib.Path(file_name).absolute()
        self._mwa_antenna_df = cudf.read_parquet(f"{path.parent}/{path.stem}_{'ANTENNA'}{path.suffix}")
        
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
            
            #path = pathlib.Path(self._cudf).absolute()
            #df = cudf.read_parquet(f"{path.parent}/{path.stem}_{'ANTENNA'}{path.suffix}")
            
            df = self._mwa_antenna_df
            station_mean = df.POSITION.compute().list.leaves.to_numpy().reshape(len(df.POSITION),3)

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


