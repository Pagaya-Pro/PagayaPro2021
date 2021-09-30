from research.utils.columns_names import COLUMNS
from tests.tests_infra.common_fixtures import *

MINIMAL_DF_SIZE = 0.75 * UNSEC_2018Q2_SAMPLE_SIZE


def test_all_unsec_2018Q2_data_samples_has_same_id(
    unsec_features,
    unsec_targets,
    unsec_targets_fixed_15,
    unsec_raw_features,
    unsec_raw_inquiry,
    unsec_trade_history,
    unsec_tradelines,
):
    """
    All the unsec_XXXX fixtures should have shared account_ids/consumer_ids.
    This is very nice for testing.
    For example when using both features and targets you can assume they have shared account ids.
    This test verify they indeed have shared ids
    """
    sample_consumer_ids = unsec_tradelines["follow_consumer_id"]
    sample_account_ids = unsec_tradelines["follow_account_id"]

    # All the ids in all the sample data frames should be included in sample_consumer_ids or sample_account_ids
    assert unsec_features[COLUMNS.SEQUENCE_NUMBER].isin(sample_consumer_ids).all()
    assert unsec_features[COLUMNS.ACCOUNT_ID].isin(sample_account_ids).all()

    assert unsec_targets[COLUMNS.ACCOUNT_ID].isin(sample_account_ids).all()
    assert unsec_targets[COLUMNS.SEQUENCE_NUMBER].isin(sample_consumer_ids).all()

    assert unsec_targets_fixed_15[COLUMNS.ACCOUNT_ID].isin(sample_account_ids).all()
    assert unsec_targets_fixed_15[COLUMNS.SEQUENCE_NUMBER].isin(sample_consumer_ids).all()

    assert unsec_raw_features["follow_consumer_id"].isin(sample_consumer_ids).all()
    assert unsec_raw_inquiry["follow_consumer_id"].isin(sample_consumer_ids).all()

    assert unsec_trade_history.index.isin(sample_account_ids).all()


def test_all_unsec_2018Q2_data_samples_has_enough_rows(
    unsec_features,
    unsec_targets,
    unsec_targets_fixed_15,
    unsec_raw_features,
    unsec_raw_inquiry,
    unsec_trade_history,
    unsec_tradelines,
):
    """
    All the unsec_XXXX fixtures should have length that is not too small (not length of 0 for example)
    I verify all the length are bigger than MINIMAL_DF_SIZE, that should be enough...
    """
    assert len(unsec_features) > MINIMAL_DF_SIZE
    assert len(unsec_targets) > MINIMAL_DF_SIZE
    assert len(unsec_targets_fixed_15) > MINIMAL_DF_SIZE
    assert len(unsec_raw_features) > MINIMAL_DF_SIZE
    assert len(unsec_raw_inquiry) > MINIMAL_DF_SIZE
    assert len(unsec_trade_history) > MINIMAL_DF_SIZE
    assert len(unsec_tradelines) > MINIMAL_DF_SIZE
