import numpy_financial
import pandas as pd
import pytest
from tests.tests_infra.snapshots_data_frames_infra import recreate_snapshots_in_file
from tests.tests_infra.testing_framework import assert_that

# Importing fixtures we will use
from tests.tests_infra.common_fixtures import unsec_targets, unsec_targets_fixed_15
# add this line so the import won't be deleted in optimize imports
_FIXTURES_USED = (unsec_targets_fixed_15, unsec_targets)


class COLUMNS:
    # Loan terms

    # The monthly scheduled payment (of an installment loan) in USD.
    MONTHLY_PAYMENT = "monthly_pmt"
    # The annual int rate, in the range 0-100. 12 * monthly_int_rate. I.e. non-compound
    INT_RATE = "int_rate"
    # The loan term in months. Normally, 36 or 60 for unsecured loans.
    TERM = "term"
    # The loan amount in USD.
    LOAN_AMOUNT = "loan_amnt"



"Read this file to learn about the assert_that testing framework for data frames"


@pytest.fixture
def dummy_targets():
    """
    A fixture that create a dummy targets for testing - include the needed columns for int_rate calculation
    """
    return pd.DataFrame(
        {
            COLUMNS.MONTHLY_PAYMENT: [160, 200],
            COLUMNS.TERM: [36, 60],
            COLUMNS.LOAN_AMOUNT: [5000, 8000],
            COLUMNS.INT_RATE: [10, 17],
        }
    )


# !!! Function To Test !!! #


def calculate_int_rate(targets):
    """
    The function we want to test
    Use numpy to calculate the loan int_rate base on its loan_amount, term and monthly_payment.
    Save the calculated int rate in "imputed_int_rate" column
    Args:
        targets: Targets to calculate int rate for
    Returns: The given targets with the calculated "imputed_int_rate" column
    """
    # calculate int_rate using numpy
    targets["imputed_int_rate"] = numpy_financial.rate(
        targets[COLUMNS.TERM], -targets[COLUMNS.MONTHLY_PAYMENT], targets[COLUMNS.LOAN_AMOUNT], 0
    )
    # move from monthly 0-1 format to yearly 0-100 format
    targets["imputed_int_rate"] = targets["imputed_int_rate"] * 1200
    return targets


# !!! Assertions on one data frame !!! #


def test_works_on_empty_df(dummy_targets):
    """
    Use:
    >> assert_that(df).XXXXX()
    to check something on a data frame.

    see assert_that doc for more information or simply continue reading this file.

    assert_that(df).is_empty() assert that the df is empty (assert len(df) == 0)

    This test check the function calculate_int_rate can handle empty data frame
    Args:
        dummy_targets: Used the fixture "dummy_targets" above
    """
    empty_targets_df = dummy_targets[dummy_targets["int_rate"] == 50]
    assert_that(empty_targets_df).is_empty()

    result = calculate_int_rate(empty_targets_df)
    assert_that(result).is_empty()


def test_imputed_int_rate_is_close_to_real_int_rate(dummy_targets):
    """
    >> assert_that(result).match_all_on_query(expr)
    Assert that all the rows in the data frame match the given query
    meaning len(df.query(expr)) == len(df)

    This test checks that the imputed int_rate calculated by calculate_int_rate is close to the real int_rate
    """
    result = calculate_int_rate(dummy_targets)
    assert_that(result).match_all_on_query("imputed_int_rate > 0.9 * int_rate & imputed_int_rate < 1.1 * int_rate")


# !!! Assertions comparing two data frames !!! #


def test_function_dont_drop_loans(dummy_targets):
    """
    Use:
    >> assert_that(actual).and_data_frame(expected).XXXX()
    to compare between data frames.

    >> assert_that(actual).and_data_frame(expected).are_same_length()
    check that both data frames are with same length (assert len(actual) == len(expected))

    This test checks that the original target has the same length as the calculate_int_rate result ->
    meaning the calculate_int_rate function doesn't drop loans
    """
    result = calculate_int_rate(dummy_targets)
    assert_that(result).and_data_frame(dummy_targets).are_same_length()


def test_dont_use_the_int_rate_column(dummy_targets):
    """
    Use:
    >> assert_that(result).and_data_frame(changed_real_int_rate_result).are_same_on_columns(["col1", "col2"])
    To check the data frames has the same values on the given columns.

    This function check the imputed_int_rate calculation don't use the original int_rate column by
    changing it and see the imputed_int_rate column was not changed
    """
    result = calculate_int_rate(dummy_targets)
    dummy_targets[COLUMNS.INT_RATE] = 30
    changed_real_int_rate_result = calculate_int_rate(dummy_targets)
    assert_that(result).and_data_frame(changed_real_int_rate_result).are_same_on_columns(["imputed_int_rate"])


# !!! Snapshot tests !!! #

recreate_snapshots_in_file(__file__)  # uncomment this line to update the snapshot


def test_the_logic_is_same_as_snapshot(dummy_targets):
    """
    Example of snapshot test.
    We use:
    >> assert_that(result).and_snapshot()...

    In the first run we should uncomment the line "recreate_snapshots_in_file(__file__)" above.
    This will save the result data frame in a file under snapshots model. The path is:
    /research/regression_data/{current_test_path}/{current_test_name}.parquet
    for this test it will be:
    tests/regression_data/tests_infra/testing_infra_tutorial/test_the_logic_is_same_as_snapshot.parquet

    In the next runs we should comment out the line "recreate_snapshots_in_file(__file__)".
    The test will open the data frame saved in the snapshot path and compare between it and the given result data frame.

    Note: If you rename the test function or the module the snapshot file and folder should be renamed to
    """
    result = calculate_int_rate(dummy_targets)
    assert_that(result).and_snapshot().are_same_shape()
    assert_that(result).and_snapshot().are_same_on_columns(["imputed_int_rate"])


def test_the_logic_is_same_as_in_two_snapshots_data_frames(dummy_targets):
    """
    If the same test saves and compare two data frames use:
    >> assert_that(result).and_snapshot(name="data_frame_name")...
    The path in that case is:
    /research/regression_data/{current_test_path}/{current_test_name}-{name}.parquet
    aka file name here is: test_the_logic_is_same_as_in_two_snapshots_data_frames-the_name_i_gave.parquet
    """
    # regression test on normal data
    result = calculate_int_rate(dummy_targets)
    assert_that(result).and_snapshot(name="result_on_normal_targets").are_same_shape()
    assert_that(result).and_snapshot(name="result_on_normal_targets").are_same_on_columns(["imputed_int_rate"])

    # regression test on data from duplicated target
    duplicated_targets = pd.concat([dummy_targets, dummy_targets])
    result_2 = calculate_int_rate(duplicated_targets)
    assert_that(result_2).and_snapshot(name="result_on_duplicated_targets").are_same_shape()
    assert_that(result_2).and_snapshot(name="result_on_duplicated_targets").are_same_on_columns(["imputed_int_rate"])


def test_the_imputed_int_rate_mean_is_close_to_snapshot(dummy_targets):
    """
    You can build a custom function that has assertions and let the framework run it with actual and snapshots.
    The function needs to get 2 data frames and has assertion. see pass_on_function docs.

    This test check the imputed_int_rates means between the current result and snapshot.
    It fails if the means of the current result is far from the mean of snapshot.
    If the rate calculation was not deterministic we would you this kind of comparision and not are_same_on_columns
    """

    def check_the_imputed_int_rate_means_are_close(actual: pd.DataFrame, expected: pd.DataFrame):
        actual_mean = actual["imputed_int_rate"].mean()
        expected_mean = expected["imputed_int_rate"].mean()
        assert abs(actual_mean - expected_mean) < 1

    result = calculate_int_rate(dummy_targets)
    assert_that(result).and_snapshot().pass_on_function(check_the_imputed_int_rate_means_are_close)


# !!! Common fixtures !!! #


def test_imputed_int_rate_is_close_to_real_int_rate_on_real_target(unsec_targets):
    """
    The common_fixtures.py module in the tests_infra folder has many useful fixtures you can import.
    In this test I imported the unsec_targets fixture from the common_fixtures module.
    In order to use a fixture just add the fixture name as a function parameter and add the import

    >> from tests.tests_infra.common_fixtures import unsec_targets

    This test is same as test_imputed_int_rate_is_close_to_real_int_rate but uses real targets data
    (sample of 100 loans).

    If you don't know what "fixture" is, see the test_hello_world.py file for pytest tutorial and exercise
    """
    result = calculate_int_rate(unsec_targets)
    assert_that(result).match_all_on_query("imputed_int_rate > 0.9 * int_rate & imputed_int_rate < 1.1 * int_rate")


def test_imputed_int_rate_on_fixed_15_targets(unsec_targets_fixed_15):
    """
    Use the unsec_targets_fixed_15 from the common_fixtures.py module
    All the int_rates in targets_fixed_15 df are 15.
    Make sure the imputed int_rate on these loans are close to 15.
    """
    result = calculate_int_rate(unsec_targets_fixed_15)
    assert 14 < result["imputed_int_rate"].mean() < 16


def test_the_logic_is_same_as_snapshot_on_real_target(unsec_targets):
    """
    Usage of snapshot testing with fixture imported from common_fixtures.py

    Note: Sample data df can weight few MB (the features sample weight around 3MB).
    When using assert_that(sample_df).and_snapshot() try to make the sample_df you pass as small as possible.
    In general use take only the columns you need for the test:
    >> assert_that(sample_df[NEEDED_COLUMNS]).and_snapshot()

    In this test we only take the imputed_int_rate columns and columns needed for the calculation
    (see needed_columns variable)
    """
    result = calculate_int_rate(unsec_targets)
    needed_columns = ["imputed_int_rate", COLUMNS.MONTHLY_PAYMENT, COLUMNS.TERM, COLUMNS.LOAN_AMOUNT, COLUMNS.INT_RATE]
    assert_that(result[needed_columns]).and_snapshot().are_same_length()
    assert_that(result[needed_columns]).and_snapshot().are_same_on_columns(["imputed_int_rate"])
