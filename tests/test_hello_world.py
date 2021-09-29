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


def test_please_test_me():
    test_string = "testing is great"
    assert please_test_me(test_string) == "testing is great!!!"


def times_7(number: Union[int, float]):
    return number * 7


# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
    assert times_7(2) == 14
    assert times_7(4) == 28
    assert times_7(0) == 0
    assert times_7(-1) == -7
    # TODO add one interesting case I didn't check
    assert times_7(float('inf')) == float('inf')
    assert times_7('h') == 'hhhhhhh'  # The function multiplies strings and not just numbers :(

@pytest.mark.parametrize('num, val', [(2,14), (0, 0), (-7, -49), ('h', 'hhhhhhh')])
def test_parametrize_times_7(num, val):
    assert times_7(num) == val

@pytest.fixture()
def numbers_times_7():
    random_generator = random.Random()
    return [random_generator.randint(-1000, 1000) for _ in range(10)]

def test_fixture_times_7(numbers_times_7):
    for num in range(len(numbers_times_7)):
        assert times_7(num) == sum([num for _ in range(7)])


# TODO Add a function and at least 3 tests
def noam_king(text):
    return text + " noam is king"

@pytest.mark.parametrize('txt', ['hi', 'idiot', 'liar'])
def test_noam_king(txt):
    assert noam_king(txt) == "{0} noam is king".format(txt)

# I will create another function and add two tests to it

def am_i_wrong(val):
    if val:
        return "Y"
    return "N"

def test_number_in_function():
    random.seed(123)
    num = random.randint(1, 10000)
    assert am_i_wrong(num) == 'Y'
    assert am_i_wrong(0) == 'N'

@pytest.mark.parametrize('val', [True, False])
def test_boolean_am_i_wrong(val):
    if val:
        assert am_i_wrong(val) == 'Y'
    else:
        assert am_i_wrong(val) == 'N'


# TODO add a function that get data frame as an argument and return it after some preprocess/change
# TODO test the function you wrote use assert_frame_equal and assert_series_equal
def modify_df(df):
    return df.dropna().reset_index(drop=True)

@pytest.fixture()
def df_series_modify_test():
    return (pd.DataFrame([['hi', np.NAN], ['what', 3], ['amigo', np.NAN]]), \
                pd.Series([1, 3, 10, 11, np.nan]))

def test_modify_df(df_series_modify_test):
    dframe, pseries = df_series_modify_test[0], df_series_modify_test[1]
    pd.testing.assert_frame_equal(modify_df(dframe), pd.DataFrame([['what', 3.0]]))
    pd.testing.assert_series_equal(modify_df(pseries), pd.Series([1.0, 3.0, 10.0, 11.0]))



def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


@pytest.mark.parametrize('x, w', [([1, 9, 8, 101], [-1, 0, 1]), ([99, 102, -3], [0])])
def test_weighted_average_raise_zero_division_error(x, w):
    # TODO check that weighted_average raise zero division error when the sum of the weights is 0
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average(x, w)