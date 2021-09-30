# from __future__ import annotations  # allow adding returned type of current class ( C.func -> C )
from typing import Callable
from typing import List
from typing import Union

import pandas as pd

from tests.tests_infra.snapshots_data_frames_infra import GLOBAL_RECREATE_FILES_LIST
from tests.tests_infra.snapshots_data_frames_infra import get_snapshot_dataframe
from tests.tests_infra.snapshots_data_frames_infra import save_dataframe_as_snapshot


def assert_that(actual: Union[pd.DataFrame]):
    """
    Use this function to check something on a data frame.
    The usage is:
    >> assert_that(df).XXXXX().XXXXX()....

    Read tests/tests_infra/testing_infra_tutorial.py to learn more on the framework

    For example:
    >> assert_that(df).is_empty()  # Check df is empty -> len(df) == 0
    >> assert_that(actual).and_data_frame(expected).are_same_length()  # Check len(actual) == len(expected)
    >> assert_that(actual).and_snapshot().are_same_length()  # Check actual is same length as snapshot df

    Args:
        actual: The data frame to check

    Returns:
        TestDataFrame object with the given data frame
    """
    return TestDataFrame(actual)


class CompareTwoDataFrames:
    """
    Syntax-Sugar that can compare two data frames (actual and expected)
    """

    def __init__(self, actual, expected):
        """
        Args:
            actual: The data frame created by the logic - the data frame "to check"
            expected: The data frame we expect to be created by the logic - the baseline we compare to

        The convention is using:
        assert_that(actual).and_data_frame(expected)....
        and NOT:
        assert_that(expected).and_data_frame(actual)....
        """
        self.actual = actual
        self.expected = expected

    @staticmethod
    def test_2_data_frames_are_the_same(actual: pd.DataFrame, expected: pd.DataFrame):
        """
        Test that the 2 data frames are the same, will check only columns for expected df and ignore dtypes
        """
        missing_columns = [c for c in expected.columns if c not in actual.columns]
        assert (
            len(missing_columns) == 0
        ), f"""
            Actual df don't have all the columns of the expected df missing columns:
            {missing_columns}
        """
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

    def are_same_on_columns(self, columns: List[str]):
        """
        Check the data frames has the same values on the given columns
        Args:
            columns: List of columns to check (ex: ["int_rate", "term"])
        """
        assert all(col in self.actual for col in columns), f"needed columns are not in actual df: {columns}"
        assert all(col in self.expected for col in columns), f"needed columns are not in expected df: {columns}"
        self.test_2_data_frames_are_the_same(self.actual[columns], self.expected[columns])

    def are_same_on_all_columns(self):
        """
        Check the data frames has the same values on all the columns in the expected data frame
        """
        self.test_2_data_frames_are_the_same(self.actual, self.expected)

    def are_completely_the_same_on_all_columns(self):
        """
        Check the 2 data frames are completely the same, include dtypes.
        This will fail often - use are_same_on_all_columns instead if you are not sure
        """
        pd.testing.assert_frame_equal(self.actual, self.expected)

    def are_same_length(self):
        """
        Check the data frames are with same length
        """
        assert len(self.actual) == len(self.expected), (
            f"length of the actual dataframe {len(self.actual)} is"
            f"different than the length of the expected dataframe "
            f"{len(self.expected)}"
        )

    def are_same_shape(self):
        """
        Check the data frames are with same shape
        """
        assert self.actual.shape == self.expected.shape, (
            "The data frames has different shapes" f"actual={self.actual.shape} " f"expected={self.expected.shape}"
        )

    def pass_on_function(self, func: Callable[[pd.DataFrame, pd.DataFrame], None]):
        """
        Run the function on the actual and expected data frames. The given function should include at least 1 assert.
        Args:
            func: The function should get the 2 data frames and include at least one assert.
                Pay attention should pass function "func" not a function call "func(..)"
        """
        func(self.actual, self.expected)


class TestDataFrame:
    """
    Syntax-Sugar object for data frame testing.
    Can assert staff like is_empty or is match_all_query
    Can return CompareTwoDataFrames with given data frame or snapshot data frame
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def and_data_frame(self, df2: pd.DataFrame) -> CompareTwoDataFrames:
        """
        Returns: An object able to compare self.db to the given df2
        """
        return CompareTwoDataFrames(actual=self.df, expected=df2)

    def and_snapshot(self, name: str = None) -> CompareTwoDataFrames:
        """
        Get snapshots or recreated it.
        Will recreate it is the test module include the line "recreate_snapshots_in_file(__file__)"
        Returns an object able to compare the tested df to snapshot
        Args:
            name: A name describing the data frame, must by used if 2 data frames are saved in the same test
        Returns: An object able to compare self.db to the snapshot data frame
        """
        if GLOBAL_RECREATE_FILES_LIST.does_current_test_should_be_recreated() and self.df is not None:
            save_dataframe_as_snapshot(name=name, present_dataframe=self.df)
            snapshot = self.df
        else:
            snapshot = get_snapshot_dataframe(name=name)
        return CompareTwoDataFrames(actual=self.df, expected=snapshot)

    def is_empty(self):
        """
        Assert data frame is empty
        """
        assert len(self.df) == 0, f"The dataframe is not empty (has length of {len(self.df)})"

    def match_all_on_query(self, expr: str):
        """
        Assert all the rows in the data frame match the given query
        meaning len(df.query(expr)) == len(df)
        """
        queried_df = self.df.query(expr)
        assert len(self.df) == len(queried_df), f"{len(self.df)-len(queried_df)} loans are filtered by the query {expr}"

