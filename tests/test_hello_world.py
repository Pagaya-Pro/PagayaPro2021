"""
Examples of tests using pytest.
Read these functions and then fix the TODOs in the end of the file.
You can learn about pytest here:
https://www.guru99.com/pytest-tutorial.html
"""
import random
from typing import Union
from typing import List
import pandas as pd
import numpy as np
import pytest


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


def test_please_test_me():
    assert please_test_me('check') == 'check!!!'


def times_7(number: Union[int, float]):
    return number * 7


# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
    assert times_7(2) == 14
    assert times_7(4) == 28
    assert times_7(0) == 0
    assert times_7(-1) == -7
    assert round(times_7(1.5),2) == 10.5
    # TODO add one interesting case I didn't check

    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])

        # assert times_7(rnd_int) > rnd_int  # TODO Explain why this assert doest work
        # It doesn't work because if rnd_int is a negative number times_7(rnd_int) will be smaller then rnd_int


@pytest.fixture()
def random_numbers():
    return [random.Random().randint(-1000, 1000) for i in range(10)]


def test_7_times_using_fixture(random_numbers):
    for i in range(10):
        assert round(times_7(random_numbers[i]),2) == sum([random_numbers[i] for j in range(7)])


@pytest.mark.parametrize('number, expected_result', [(2, 14), (4, 28), (0, 0), (-1, -7), (1.5, 10.5)])
def test_7_times_using_parametrize(number, expected_result):
    assert times_7(number) == expected_result


def calculate_sqrt(number: Union[int, float]):
    if number < 0:
        raise Exception("Can't do sqrt on negative number")
    return number ** 0.5


def test_calculate_sqrt_negative():
    with pytest.raises(Exception):
        assert calculate_sqrt(-2)


def test_calculate_sqrt():
    assert calculate_sqrt(4) == 2


def test_calculate_sqrt_float():
    assert round(calculate_sqrt(10.5), 2) == 3.24


# TODO Add a function and at least 3 tests

# TODO add a function that get data frame as an argument and return it after some preprocess/change
# TODO test the function you wrote use assert_frame_equal and assert_series_equal

def change_df(df: pd.DataFrame):
    df['value'] = 5.0
    return df


def change_series(ser: pd.Series):
    return ser.head(1)


@pytest.fixture
def data_frame_example():
    data = dict(
        name=['Tal', 'Ron', 'Amit'],
        value=[1, 5, np.nan]
    )
    return pd.DataFrame(data)


@pytest.fixture
def data_frame_expected():
    return pd.DataFrame(dict(
        name=['Tal', 'Ron', 'Amit'],
        value=[5.0, 5.0, 5.0]))


@pytest.fixture
def series_example():
    data = [1, 5, 3]
    return pd.Series(data)


@pytest.fixture
def series_expected():
    data = [1]
    return pd.Series(data)


def test_change_series(series_example, series_expected):
    pd.testing.assert_series_equal(change_series(series_example), series_expected)


def test_change_df(data_frame_example, data_frame_expected):
    pd.testing.assert_frame_equal(change_df(data_frame_example), data_frame_expected)


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    with pytest.raises(ZeroDivisionError):
        compute_weighted_average([1], [0])
