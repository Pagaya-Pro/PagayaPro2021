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


# TODO test this function, make sure for example please_test_me("testing is great") = "testing is great!!!" - done
def please_test_me(string: str) -> str:
    return string + "!!!"


def test_please_test_me():
    assert isinstance(please_test_me('hi'), str)
    assert please_test_me("testing is great") == "testing is great!!!"
    assert please_test_me("!!!") == "!!!!!!"


def times_7(number: Union[int, float]):
    return number * 7


@pytest.mark.parametrize("num, num_times_7", [(2, 14),(4, 28),(0, 0),(-1, -7),(0.1, 0.7)])
def test_times_7_use_parametrize(num, num_times_7):
    assert times_7(num) == num_times_7


@pytest.fixture
def generate_random_nums():
    """generate 10 random integers"""
    random_generator = random.Random()
    return [random_generator.randint(-1000, 1000) for i in range(10)]


def test_times_7_use_fixture(generate_random_nums):
    arr = generate_random_nums()
    for rnd_int in arr:
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])

# TODO Add a function and at least 3 tests - done
def factorial(a):
    """calculates factorial of an integer"""
    if a < 1 or a != int(a):
        return -1
    if a == 1:
        return 1
    return a * factorial(a-1)


@pytest.mark.parametrize("num", [0,0.1,-3])
def test_factorial_smaller_than_1(num):
    assert factorial(num) == -1


@pytest.mark.parametrize("num, fact", [(3, 6),(4, 24),(1, 1)])
def test_factorial_legit(num, fact):
    assert factorial(num) == fact


@pytest.mark.parametrize("num", [1.1,1.8,13.9])
def test_factorial_not_an_integer(num):
    assert factorial(num) == -1


# TODO add a function that get data frame as an argument and return it after some preprocess/change - done
def keep_even_rows(df: pd.DataFrame):
    """keeps only even rows of DF"""
    return df.iloc[::2].reset_index(drop=True)


# TODO test the function you wrote use assert_frame_equal and assert_series_equal - done
def test_keep_even_rows():
    df = pd.DataFrame({'a' : [1,2,3,4], 'b' : [5,6,7,8], 'c' : [9,10,11,12]})
    df_even = pd.DataFrame({'a' : [1, 3], 'b' : [5,7], 'c' : [9, 11]})
    pd.testing.assert_frame_equal(keep_even_rows(df), df_even, check_less_precise=True)

    ser = pd.Series([1,2,3,4])
    ser_even = pd.Series([1,3])
    pd.testing.assert_series_equal(keep_even_rows(ser), ser_even)


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    """check that exception is raised when w sums to 0"""
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([1,2,3,4], [1,1,-1,-1])
    # TODO check that weighted_average raise zero division error when the sum of the weights is 0 - done
