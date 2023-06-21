import dask
import dask.array as da
import numpy as np
from dask.distributed import Client
from time import perf_counter
from daskms import xds_from_ms
from casacore import tables
import pypeline.phased_array.measurement_set as measurement_set
import pypeline.phased_array.data_gen.statistics as vis

ms_file = "/work/backup/ska/gauss4/gauss4_t201806301100_SBL180.MS"

columns=["TIME", "ANTENNA1", "ANTENNA2", "FLAG", "DATA"]


def dataset_row_chunks(ds, cutoff=10000):
        """
        Given a desired number of rows, the rows for each unique time
        are grouped together into a chunk until this limit is exceeded.
        Then, the next chunk starts until no more unique times are left.

        This achieves two purposes:

        1. Load a significant amount of data in a single IO operation by
           reading many rows at once.
        2. Related to (1), gives each dask thread a significant amount
           of work to do.
        2. Ensures that the data for each unique time is stored in a
           single chunk (i.e. not split across multiple chunks)

        Parameters
        ----------
        datasets : list of datasets

        Returns
        -------
        row_chunks : list of tuples
            Final row chunking schema for each dataset.
        """
        unique_values, count = da.unique(ds.TIME.data, return_counts=True)
        unique_time, counts = dask.compute(unique_values, count)

        row_chunks = []

        row_sum, dataset_rows = 0, []

        for t, c in zip(unique_time, counts):
            if row_sum + c < cutoff:
                row_sum += c
            elif row_sum > 0:
                dataset_rows.append(row_sum)
                row_sum = c
            else:
                raise ValueError(f"Count {c} for time {t} is > "
                                 f"--desired-rows={cutoff}"
                                 f"Increase desired rows")

        if row_sum > 0:
            dataset_rows.append(row_sum)

        row_chunks.append(tuple(dataset_rows))

        return row_chunks
    
def visibility(time, ant1, ant2, data, flag, beam_id, channel_id=[0]):

    utime, idx, cnt = np.unique(time, return_index=True, return_counts=True)
    
    t_return = []
    vs_return = []
    missing_return = []

    for i in range(1):
        tobs = utime[i]*1.15741e-5
        start=idx[i]
        end=start+cnt[i]
        beam_id_0 = ant1[start:end]
        beam_id_1 = ant2[start:end]
        data_flag = flag[start:end]
        data = data[start:end]
        
        # We only want XX and YY correlations
        data = np.average(data[:, :, [0, 3]], axis=2)[:, channel_id]
        data_flag = np.any(data_flag[:, :, [0, 3]], axis=2)[:, channel_id]
        # data[data_flag] = 0
        S_full_idx = pd.MultiIndex.from_arrays((beam_id_0, beam_id_1), names=("B_0", "B_1"))
        S_full = pd.DataFrame(data=data, index=S_full_idx)
        
        # Drop rows of `S_full` corresponding to unwanted beams.
        #beam_id = np.unique(self.instrument._layout.index.get_level_values("STATION_ID"))
        N_beam = len(beam_id)
        i, j = np.triu_indices(N_beam, k=0)
        wanted_index = pd.MultiIndex.from_arrays((beam_id[i], beam_id[j]), names=("B_0", "B_1"))
        index_to_drop = S_full_idx.difference(wanted_index)
        S_trunc = S_full.drop(index=index_to_drop)

        index_diff = wanted_index.difference(S_trunc.index)
        
        missing = index_diff.to_numpy()
        
        # Break S into columns and stream out
        beam_idx = pd.Index(beam_id, name="BEAM_ID")
        v = measurement_set._series2array(S_full[0].rename("S", inplace=True))
        visibility = vis.VisibilityMatrix(v, beam_idx)
        t_return.append(tobs)
        vs_return.append(visibility)
        missing_return.append(missing)
        
        
    return t_return, vs_return, missing_return
        
        
if __name__ == "__main__":
    
    ms_file = "/work/backup/ska/gauss4/gauss4_t201806301100_SBL180.MS"
    
    ms = measurement_set.LofarMeasurementSet(ms_file, N_station=37, station_only=True)
    beam_ids = np.unique(ms.instrument._layout.index.get_level_values("STATION_ID"))

    ds = xds_from_ms(ms_file, columns=["TIME"])[0]

    row_chunks = dataset_row_chunks(ds, cutoff=10000)
    
    datasets = xds_from_ms(ms_file, columns=columns, chunks=[{"row": rc for rc in row_chunks}])[0]
    
    x=datasets.compute()
    
    time = x.TIME.data
    ant1 = x.ANTENNA1.data
    ant2 = x.ANTENNA2.data
    flag = x.FLAG.data
    data = x.DATA.data 

    tt, vv, mm = visibility(time, ant1, ant2, data, flag, beam_ids, channel_id=[0])
