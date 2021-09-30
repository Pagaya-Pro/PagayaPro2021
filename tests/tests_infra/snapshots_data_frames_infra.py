import os
import pathlib

import pandas as pd

from tests import TESTING_DIR_PATH

REGRESSION_DATA = TESTING_DIR_PATH / "data" / "snapshots"

PAST_PATH_NOT_EXIST_ERROR_MESSAGE = """
    Could not find the snapshot df. The path {snapshot_path} does not exists.
    This could happen because of few reasons
    1.  This is the first time you run the test and the file was not created yet.
        To fix this write this line:
        >> RecreateFilesList.append(__file__)
        in the test module and run the test again, this will create the needed files.
        Then comment it out and try again.

    2.  You didn't commit the files to the repo and the test fails on github.
        Add all the files in {snapshot_path} folder to git, commit them and push.
        The test in github should restart and pass.

    3.  You run the test on azure/aws so files were created there. You don't have them locally and didn't commit them.
        To fix this download the files from azure/aws:
        Right on {snapshot_path} folder -> Deployment -> Download from AWS/Azure
        Then add the files to git and commit them

    4.  You renamed the test module.
        Find the folder with the old module name in regression_data and rename it.

    5.  You moved the test module.
        Find the folder with the module name and move it to the right location.

    6.  You renamed the test function.
        Look on the files in the folder {snapshot_path} and rename the files with the old test function name to the new one.
"""


class RecreateFilesList:
    """
    A class that maintain a list of file.
    When running snapshot tests we should recreated the snapshots for all the files in the list.
    Use:
    >> RecreateFilesList.append(__file__)
    to recreate the snapshot for all the tests in the file.

    This class really usage is via recreate_snapshots_in_file function below
    """

    def __init__(self):
        self.recreate_list = []

    def append(self, file_path: str):
        """
        Add the given file_path to the test modules to recreate the snapshot in
        """
        self.recreate_list.append(pathlib.Path(file_path).name)

    def does_current_test_should_be_recreated(self):
        """
        Returns: True if the snapshot of the pytest test that is running now should be recreated
        """
        current_test_path = os.environ.get("PYTEST_CURRENT_TEST").split("::")[0]
        test_file_name = pathlib.Path(current_test_path).name
        return test_file_name in self.recreate_list


GLOBAL_RECREATE_FILES_LIST = RecreateFilesList()


def recreate_snapshots_in_file(file_path: str):
    GLOBAL_RECREATE_FILES_LIST.append(file_path)


def get_snapshot_path(name: str = None):
    """
    The path path is:
    research/regression_data/{current_test_path}/{current_test_name}-{name}.parquet

    1. Use current_test_path not including the folder "tests" and ".py"
    2. If name is None use file name f"{current_test_name}.parquet"
    """
    current_test_path, current_test_name = os.environ.get("PYTEST_CURRENT_TEST").split("::")
    current_test_name = current_test_name.split(" ")[0]
    test_module_data_dir = REGRESSION_DATA / current_test_path[len("tests/") : -len(".py")]
    if name is None:
        return test_module_data_dir / f"{current_test_name}.parquet"
    return test_module_data_dir / f"{current_test_name}-{name}.parquet"


def get_snapshot_dataframe(name: str):
    """
    Read and return the data frame saved as a snapshot.
    Raises:
         AssertionError If snapshot path does not exist.
    """
    snapshot_path = get_snapshot_path(name)
    if not snapshot_path.exists():
        raise FileNotFoundError(PAST_PATH_NOT_EXIST_ERROR_MESSAGE.format(snapshot_path=snapshot_path))

    print(f"find snapshot data frame in {snapshot_path}")
    return pd.read_parquet(snapshot_path)


def save_dataframe_as_snapshot(name: str, present_dataframe: pd.DataFrame):
    """
    Save the given present_dataframe in the snapshot_path
    """
    snapshot_path = get_snapshot_path(name)
    print(f"Saved the current parquet to the tests modules in path {snapshot_path}")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    present_dataframe.to_parquet(snapshot_path)
