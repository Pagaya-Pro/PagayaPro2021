from typing import List
import pandas as pd
import math
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm
import psutil


def get_recommended_cores_amount(number_of_jobs_to_run: int = None, use_virtual_cores: bool = False) -> int:
    """

    Args:
        number_of_jobs_to_run: Number of jobs needed.
        i.e: If you got only 5 jobs to parallelize, run:
        >>>>get_recommended_cores_amount(number_of_jobs_to_run=5)
        If the recommended number of cores willl be bigger it will return 5.
        use_virtual_cores: To use the virtual cores too, if needed.
    Returns: Recommended number of cores to use

    """
    # each physical cores is 2 virtual cores
    cores_in_computer = psutil.cpu_count(logical=use_virtual_cores)
    if cores_in_computer > 30:
        max_cores_to_use = cores_in_computer - 2  # leave some cores for other processes
    elif cores_in_computer > 4:
        max_cores_to_use = cores_in_computer - 1  # leave one core for other processes
    else:
        max_cores_to_use = cores_in_computer
    if number_of_jobs_to_run is None:
        return max_cores_to_use
    else:
        return min(number_of_jobs_to_run, max_cores_to_use)


class TqdmParallel(Parallel):
    def __init__(self, desc="", size=None, leave=False, *arg, **kwarg):
        """
        A Parallel object that prints the threads progress -> how many threads were finished out of the total.
        Args:
            desc: Description to print
            size: Give the jobs amount if the iterable you will give in the __call__ func will be generator
                 (wont have __len__)
            leave: If False will delete the process bar after work is finished
            *arg: Args to give the Parallel class
            **kwarg: Kwargs to give the Parallel class
        """
        self.desc = desc
        self.size = size
        self.leave = leave
        self._pbar = None
        Parallel.__init__(self, *arg, **kwarg)

    def __call__(self, iterable):
        """
        Args:
            iterable: Same as joblib.Parallel
        """
        size = self.size or len(iterable)
        with tqdm(total=size, desc=self.desc, leave=self.leave) as self._pbar:
            return Parallel.__call__(self, iterable)

    def print_progress(self):
        """
        Overrides inner joblib.Parallel method. Update the progress bar.
        """
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def is_s3_path(path):
    return path.startswith("s3")


def s3_glob(s3_path, pattern):
    if not is_s3_path(s3_path):
        raise ValueError('Only s3 uri paths are supported.')

    return [f"s3://{key}" for key in s3fs.S3FileSystem().glob(f"{s3_path}/{pattern}")]


def parquet_to_dataframe(
        path: str,
        columns: List[str] = None,
        frac: float = 1.0,
        files_pattern: str = "*.parquet",
        **read_parquet_kwargs,
) -> pd.DataFrame:
    """
    reads parquet files into dataframe efficiently.

    single file =>    no parallelization, no disk IO
    multiple files => download to files in parallel to tmp files,and concat them

    Args:
        path: parquet file or directory containing parquet files
        columns : Optional, If not None, only these columns will be
                  read from the file.
        frac: the fraction of files to read in case path is a directory of parquets
        files_pattern: a pattern of files to read from the directory, in case path is a directory of parquets
        **read_parquet_kwargs: see `pyarrow.parquet.read_table`
    """
    assert 0 < frac <= 1
    assert not files_pattern or "**" not in files_pattern, "multiple levels are not supported"

    partitions_paths = sorted(s3_glob(path, files_pattern))
    if len(partitions_paths) == 0:
        partitions_paths = [path]

    num_of_files_to_read = math.ceil(frac * len(partitions_paths))
    partitions_paths = partitions_paths[:num_of_files_to_read]
    n_jobs = get_recommended_cores_amount(num_of_files_to_read)
    dfs = TqdmParallel(n_jobs=n_jobs, size=num_of_files_to_read,
                       desc=f"read {num_of_files_to_read} files using {n_jobs} cores... ", leave=True)(
        delayed(pd.read_parquet)(p) for p in partitions_paths)
    return pd.concat(dfs, axis=0)