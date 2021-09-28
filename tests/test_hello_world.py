"""
Examples of tests using pytest.
Read these functions and then fix the TODOs in the end of the file.
You can learn about pytest here:
https://www.guru99.com/pytest-tutorial.html
"""
import random
from typing import Union
from typing import List

import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("word", [3, ""])
def test_please_test_me(word):
    assert isinstance(please_test_me(word), str), "Please use strings."
    assert len(please_test_me(word)) == len(word) + 3


def times_7(number: Union[int, float]):
    return number * 7


# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
    assert times_7(2) == 14
    assert times_7(4) == 28
    assert times_7(0) == 0
    assert times_7(-1) == -7
    # TODO add one interesting case I didn't check

    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_int) == sum([rnd_int for _ in range(7)])

        # assert times_7(rnd_int) > rnd_int  # TODO Explain why this assert doest work
        # negative numbers / zero


@pytest.mark.parametrize("number, times_seven", [(0, 0), (-1, -7), (2, 14), (4, 28), (np.Inf, np.inf)])
def test_times_7_param(number, times_seven):
    assert times_7(number) == times_seven


def mult_by_sum(num, how_many):
    """
    This calculates multiplication by summation.
    """
    return sum([num for _ in range(how_many)])


@pytest.fixture
def rnd_int_list():
    random_generator = random.Random()
    rnd_int_list = [random_generator.randint(-1000, 1000) for _ in range(10)]
    return rnd_int_list


def test_times_7_fixture(rnd_int_list):
    # using an implementation of multiplication by summation through a fixture.
    for rnd_int in rnd_int_list:
        assert times_7(rnd_int) == mult_by_sum(rnd_int, 7)


# TODO Add a function and at least 3 tests
def normalize(arr):
    '''
    Args:
        arr: 1D array of numbers

    Returns:
        Normalized array
    '''
    std = np.std(arr)
    if std == 0:
        std = 1
    return (arr - np.mean(arr)) / std


@pytest.fixture(scope='module')
def curr_arr():
    return [1, 1, 1, 1]


def test_normalize_empty(curr_arr):
    assert len(curr_arr) != 0, "Please check that the array is not empty."


def test_normalize_mean(curr_arr):
    # Round in case it doesn't recognize it as 0, we have enough sign' digits.
    assert round(np.mean(normalize(curr_arr)),10) == 0, "Please check infs"


def test_normalize_std(curr_arr):
    # Round in case it doesn't recognize it as 0, we have enough sign' digits.
    assert round(np.std(normalize(curr_arr)),10) == 1, "Please use an array with more than 1 unique value."


# TODO add a function that get data frame as an argument and return it after some preprocess/change
def total_siblings_df(df):
    '''
    Args:
        df: Dataframe
    Returns: col1 without missing values, unless both cols have missing values, then return an empty df.
    '''
    df['siblings'] = df.brothers + df.sisters
    return df


# TODO test the function you wrote use assert_frame_equal and assert_series_equal
@pytest.fixture
def siblings_frame():
    data = dict(
        brothers=[2, 2, 3, 4, 5, 4, 3, 2, 1],
        sisters=[0, 0, 0, 4, 5, 4, 3, 2, 1],
    )
    return pd.DataFrame.from_dict(data, orient='columns')


def test_siblings(siblings_frame):
    full_siblings_frame = total_siblings_df(siblings_frame)
    # Test whole numbers
    pd.testing.assert_frame_equal(full_siblings_frame, full_siblings_frame.astype(int)), \
        'inputs should be whole numbers.'
    # Test negative numbers / inf
    bool_siblings = (full_siblings_frame['siblings'] >= 0)
    bool_test = pd.Series(np.ones(len(bool_siblings,)).astype(bool))
    pd.testing.assert_series_equal(bool_siblings, bool_test, check_names=False), \
        'Total siblings containes negative or nan values'


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([1, 2, 1], [-2, 2, 0])
