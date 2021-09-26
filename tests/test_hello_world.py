"""
Examples of tests using pytest.
Read these functions and then fix the TODOs in the end of the file.
You can learn about pytest here:
https://www.guru99.com/pytest-tutorial.html
"""
import random
from typing import Union
from typing import List

import pytest
import pandas as pd
import numpy as np
import math


def test_simple_test():
    assert 1 > 0
    assert len("123456") == 6
    assert 2 > 1 * 1.5, "This will be print if the test fails"


def fibonacci(n: int) -> int:
    """
    Will return the Fn element of fibonacci series
    Args:
        n: The element index

    Returns:
        The n'th fibonacci number
    """
    if n < 2:
        return 1
    fn = 1
    fn_minus_1 = 1
    for i in range(2, n + 1):
        next_item = fn + fn_minus_1
        fn_minus_1 = fn
        fn = next_item
    return fn


# Test the function using parametrize. Give few examples of index and the fibonacci element in that index.
# learn more about parametrize here: https://www.guru99.com/pytest-tutorial.html#11
@pytest.mark.parametrize("item_index, fibonacci_value", [(0, 1), (1, 1), (2, 2), (4, 5), (6, 13), (8, 34)])
def test_fibonacci_using_parametrize(item_index, fibonacci_value):
    assert fibonacci(item_index) == fibonacci_value


# Test the function using fixture.
# learn more about fixture here: https://www.guru99.com/pytest-tutorial.html#10


@pytest.fixture
def first_fibonacci_numbers():
    """
    This fixture is the first elements of the fibonacci series.
    (In real life this better be a constant, we use fixture for generating objects we need for testing)
    """
    return [1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_fibonacci_using_fixture(first_fibonacci_numbers):
    """
    Test the fibonacci function. Tests the first elements of the series.
    Args:
        first_fibonacci_numbers: This is a fixture so it is automatically filled.
            The first_fibonacci_numbers will have the first elements of the fibonacci series
            see first_fibonacci_numbers() function.
    """
    for item_index, fibonacci_value in enumerate(first_fibonacci_numbers):
        assert fibonacci(item_index) == fibonacci_value


# TODO test this function, make sure for example please_test_me("testing is great") = "testing is great!!!"
def please_test_me(string: str) -> str:
    return string + "!!!"


@pytest.mark.parametrize("orig_str, expected_str", [("testing is great", "testing is great!!!"),
                                                    ("testing hello world", "testing hello world!!!")])
def test_please_test_me(orig_str, expected_str):
    assert please_test_me(orig_str) == expected_str


def times_7(number: Union[int, float]):
    return number * 7


# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
    assert times_7(2) == 14
    assert times_7(4) == 28
    assert times_7(0) == 0
    assert times_7(-1) == -7
    # TODO add one interesting case I didn't check
    assert times_7(1) == 7

    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])

        # assert times_7(rnd_int) > rnd_int  # TODO Explain why this assert doest work
        # Explain: for negative values this assumption isn't true


@pytest.mark.parametrize("num, num_times_7", [(2, 14), (4, 28), (0, 0), (-1, -7), (1, 7)])
def test_times_7_parametrize(num, num_times_7):
    assert times_7(num) == num_times_7


@pytest.fixture
def random_number():
    return random.Random().randint(-1000, 1000)


def test_times_7_fixture(random_number):
    for i in range(10):
        assert times_7(random_number) == sum([random_number for i in range(7)])


# TODO Add a function and at least 3 tests
def absolute_value(number: Union[int, float]):
    return math.sqrt(pow(number, 2))


@pytest.mark.parametrize("num", [1, 2, 5.5, 100])
def test_positive_absolute_value(num):
    assert absolute_value(num) == num


@pytest.fixture
def negative_value():
    return random.Random().randint(-1000, -1)


def test_negative_integer_absolute_value(negative_value):
    assert absolute_value(negative_value) == 0 - negative_value


@pytest.mark.parametrize("num", [-1.1, -7.8, -99.9])
def test_negative_float_abolute_value(num):
    assert absolute_value(num) == 0 - num


def test_zero_absolute_value():
    assert absolute_value(0) == 0


# TODO add a function that get data frame as an argument and return it after some preprocess/change
# TODO test the function you wrote use assert_frame_equal and assert_series_equal
def add_ones_column(df: pd.DataFrame) -> pd.DataFrame:
    df['ones'] = np.ones(df.shape[0])
    return df


@pytest.fixture
def data_frame_example():
    data = dict(
        person_name=["Ron", "Roy", "Shai", "Yuval"],
        values=[1, 4, 3, 2]
    )
    return pd.DataFrame(data)


def test_add_ones_column(data_frame_example):
    tested_df = add_ones_column(data_frame_example)
    expected_df = pd.DataFrame(dict(
        person_name=["Ron", "Roy", "Shai", "Yuval"],
        values=[1, 4, 3, 2],
        ones=[1., 1., 1., 1.])
    )
    pd.testing.assert_frame_equal(tested_df, expected_df)
    pd.testing.assert_series_equal(tested_df.ones, expected_df.ones)


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    # TODO check that weighted_average raise zero division error when the sum of the weights is 0
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([1, 2, 3], [-1, 0, 1])
