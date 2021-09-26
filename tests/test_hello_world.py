"""
Examples of tests using pytest.
Read these functions and then fix the TODOs in the end of the file.
You can learn about pytest here:
https://www.guru99.com/pytest-tutorial.html
"""
import math
import random
from typing import Union
from typing import List

import pytest
import pandas as pd

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


# VTODO test this function, make sure for example please_test_me("testing is great") = "testing is great!!!"
def please_test_me(string: str) -> str:
    return string + "!!!"

def test_please_test_me():
    assert please_test_me("testing is great") == "testing is great!!!"

def times_7(number: Union[int, float]):
    return number * 7


# VTODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
    assert times_7(2) == 14
    assert times_7(4) == 28
    assert times_7(0) == 0
    assert times_7(-1) == -7
    # VTODO add one interesting case I didn't check
    assert times_7(7) == 7 ** 2

    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])

        # assert times_7(rnd_int) > rnd_int  # VTODO Explain why this assert doest work
        # not true in the negative value case ie times_7(-1) = -7 < -1 = rnd_int

@pytest.mark.parametrize(("x","y"), [(2,14),(4,28),(0,0),(-1,-7),(7,49)])
def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize(x,y):
    assert times_7(x) == y

@pytest.fixture()
def someFixture():
    class one_unit(object):
        param = 10
        res = 70
    return one_unit

def test_make_me_fixture(someFixture):
    assert times_7(someFixture.param) == someFixture.res

# VTODO Add a function and at least 3 tests
def getMaxOfArr(arr):
    if arr==[]:
        return -math.inf
    return max(arr[0], getMaxOfArr(arr[1:]))
def test_normal_arr_getMaxOfArr():
    assert getMaxOfArr([-3,3,44,-435,63636]) == 63636
def test_empty_arr_getMaxOfArr():
    assert getMaxOfArr([]) == -math.inf
def test_floats_arr_getMaxOfArr():
    assert getMaxOfArr([1.222,1.221,0.999, -0.00009, 1.22200000001]) == 1.22200000001
# VTODO add a function that get data frame as an argument and return it after some preprocess/change
def resetAndDropDF(df):
    df_c = df.copy()
    df_c = df_c.reset_index()
    return df_c.dropna()
# VTODO test the function you wrote use assert_frame_equal and assert_series_equal
def test_resetAndDropDF():
    data_before = {'product_name': ['laptop', 'printer', 'tablet', 'desk', 'chair', pd.NA],
            'price': [1200, 150, 300, 450, 200, pd.NA]
            }

    df_before = pd.DataFrame(data_before).set_index('product_name')

    df_after = df_before.reset_index().dropna()
    pd.testing.assert_frame_equal(resetAndDropDF(df_before), df_after)
    pd.testing.assert_series_equal(resetAndDropDF(df_before)['product_name'],df_after['product_name'])

def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([1, 2, 3, 4], [1, -1, 1, -1])
# VTODO check that weighted_average raise zero division error when the sum of the weights is 0