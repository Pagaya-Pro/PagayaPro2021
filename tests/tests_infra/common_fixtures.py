from typing import Dict

import pandas as pd
import pytest

from tests import TESTING_DIR_PATH

STANDARD_DATA_FOLDER = TESTING_DIR_PATH / "data" / "standard_data"
UNSEC_2018Q2_SAMPLE_SIZE = 100


@pytest.fixture
def experiment_data_before_updaters() -> Dict:
    """
    Unsec experiment data before running targets updater and features updates.
    Good for testing updaters and experiment infra logic.
    Returns:
        A Dataset with feature and targets after most of the experiment preprocessing and before the updaters
        Include 100 loans
    """
    return dict(
        features=pd.read_parquet(STANDARD_DATA_FOLDER / "experiment_data_before_updaters_features.parquet"),
        targets=pd.read_parquet(STANDARD_DATA_FOLDER / "experiment_data_before_updaters_targets.parquet"),
    )


#####   Data Taken from UNSEC_0_2018Q2_PARAM_QUARTER_DOCUMENTATION   ####
# The ids among the data frames are the same. For example features, targets and tradelines will have same costumers id
# This is very nice for testing.
# For example when using both features and targets you can assume they have shared account ids.
#########################################################################


def get_unsec_2018Q2_sample_file_path(data_frame_type: str):
    return STANDARD_DATA_FOLDER / f"unsec_0_2018Q2_sample_100_{data_frame_type}.parquet"


@pytest.fixture
def unsec_features() -> pd.DataFrame:
    """
    Unsec features sample taken from UNSEC_0_2018Q2_PARAM_QUARTER_DOCUMENTATION.reg_features_with_income
    """
    return pd.read_parquet(get_unsec_2018Q2_sample_file_path("features"))


@pytest.fixture
def unsec_targets() -> pd.DataFrame:
    """
    Unsec targets sample taken from UNSEC_0_2018Q2_PARAM_QUARTER_DOCUMENTATION.targets
    """
    return pd.read_parquet(get_unsec_2018Q2_sample_file_path("targets"))


@pytest.fixture
def unsec_targets_fixed_15() -> pd.DataFrame:
    """
    Unsec targets with fixed int rate 15 sample taken from UNSEC_0_2018Q2_PARAM_QUARTER_DOCUMENTATION.targets_fixed
    """
    return pd.read_parquet(get_unsec_2018Q2_sample_file_path("targets_fixed_15"))


@pytest.fixture
def unsec_raw_features() -> pd.DataFrame:
    """
    Unsec raw features sample taken from UNSEC_0_2018Q2_PARAM_QUARTER_DOCUMENTATION.raw_features
    """
    return pd.read_parquet(get_unsec_2018Q2_sample_file_path("raw_features"))


@pytest.fixture
def unsec_raw_inquiry() -> pd.DataFrame:
    """
    Unsec raw inquiry sample taken from UNSEC_0_2018Q2_PARAM_QUARTER_DOCUMENTATION.raw_inquiries
    """
    return pd.read_parquet(get_unsec_2018Q2_sample_file_path("raw_inquiry"))


@pytest.fixture
def unsec_trade_history() -> pd.DataFrame:
    """
    Unsec trade history sample taken from UNSEC_0_2018Q2_PARAM_QUARTER_DOCUMENTATION.raw_trade_history
    Pay attention: the trade history here is after reading using read_prama_trade_history_tar_file
    meaning it has account id as index and only the monthly payment
    """
    return pd.read_parquet(get_unsec_2018Q2_sample_file_path("trade_history"))


@pytest.fixture
def unsec_tradelines() -> pd.DataFrame:
    """
    Unsec raw tradelines sample taken from UNSEC_0_2018Q2_PARAM_QUARTER_DOCUMENTATION.raw_tradelines
    """
    return pd.read_parquet(get_unsec_2018Q2_sample_file_path("tradelines"))

