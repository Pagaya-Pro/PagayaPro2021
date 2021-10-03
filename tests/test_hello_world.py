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
import pytest
from pandas._testing import assert_series_equal


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



def please_test_me(string: str) -> str:
    return string + "!!!"

def test_please_test_me():
    assert please_test_me("testing is great") == "testing is great!!!"


def times_7(number: Union[int, float]):
    return number * 7



def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
    assert times_7(2) == 14
    assert times_7(4) == 28
    assert times_7(0) == 0
    assert times_7(-1) == -7
    assert times_7(0.5) == 3.5

@pytest.mark.parametrize("num", [2, 4, 0, -1, 0.5])

def test_times_7_using_param(num):
    assert times_7(num) == sum([num for i in range(7)])

@pytest.fixture
def num_gen():
    random_generator = random.Random()
    ret_arr = []
    for i in range(10):
        ret_arr.append(random_generator.randint(-1000, 1000))
    return ret_arr


def test_times_7_using_fixture(num_gen):
    for num in num_gen:
        assert times_7(num) == sum([num for i in range(7)])


        # assert times_7(rnd_int) > rnd_int  # For negative inputs, the returned value will be smaller than the input, which is fine, but the test will fail.


def boolean_xor(first, second):
    return ((not first) and second) or (first and (not second))

def test_boolean_xor_using_bools():
    assert boolean_xor(True, True) == False
    assert boolean_xor(True, False) == True
    assert boolean_xor(False, True) == True
    assert boolean_xor(False, False) == False

def test_boolean_xor_using_binary():
    assert boolean_xor(1, 1) == 0
    assert boolean_xor(1, 0) == 1
    assert boolean_xor(0, 1) == 1
    assert boolean_xor(0, 0) == 0

def test_boolean_xor_using_ints():
    assert boolean_xor(22, 53) == 0
    assert boolean_xor(156, 0) == 1
    assert boolean_xor(0, 0) == 0


def add_22_to_df_ints(df):
    return df.sum(axis=0)

def test_add_22_to_df_ints():
    nums = [[1,2,3,4], [5,6,7,8]]
    cols = ['first', 'second', 'third', 'fourth']
    new_df = pd.DataFrame(nums, columns=cols)
    new_ser = pd.Series(index=['first', 'second', 'third', 'fourth'], data=[6,8,10,12])
    assert_series_equal(add_22_to_df_ints(new_df), new_ser)


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    with pytest.raises(ZeroDivisionError):
        assert(compute_weighted_average([1,2,3],[-1,0,1]))