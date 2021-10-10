"""
Examples of tests using pytest.
Read these functions and then fix the TODOs in the end of the file.
You can learn about pytest here:
https://www.guru99.com/pytest-tutorial.html
"""
import random
from typing import Union, List
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
    assert please_test_me("wow!!!") == "wow!!!!!!"


def times_7(number: Union[int, float]):
    return number * 7


# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
@pytest.mark.parametrize("a, a_times_7", [(2,14), (4,28), (0,0), (-1,-7), (1/2,7/2)])
def test_times_7_with_parametrize(a, a_times_7):
    assert times_7(a) == a_times_7

@pytest.fixture
def generate_test_numbers():
    random_generator = random.Random()
    return [random_generator.randint(-1000, 1000) for i in range(10)]

def test_times_7_with_fixture(generate_test_numbers):
    for rnd_int in generate_test_numbers:
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])
        # assert times_7(rnd_int) > rnd_int  # TODO Explain why this assert doest work
        # Won't work because it isn't enough for the result to be greater than the parameter.
        # For example, 6>5 but 6 isn't 5 times 7.


# TODO Add a function and at least 3 tests
def subtract_3(x):
    return x-3

def test_subtract_3():
    assert subtract_3(10) == 7

@pytest.mark.parametrize("a, a_minus_3", [(3,0), (0,-3), (1003.5, 1000.5), (-34, -37)])
def test_subtract_3_with_parametrize(a, a_minus_3):
    assert subtract_3(a) == a_minus_3

@pytest.fixture
def generate_test_numbers():
    random_generator = random.Random()
    return [random_generator.randint(-1000, 1000) for i in range(10)]

def test_subtract_3_with_fixture(generate_test_numbers):
    for rnd_int in generate_test_numbers:
        assert subtract_3(rnd_int)+3 == rnd_int

# TODO add a function that get data frame as an argument and return it after some preprocess/change
def rename_first_col(df, new_name):
    old_name = df.columns[0]
    return df.rename(columns={old_name: new_name})

# TODO test the function you wrote use assert_frame_equal and assert_series_equal
@pytest.fixture
def dfs_to_test():
    df = pd.DataFrame([[1,2,3],[4,5,6]])
    df.columns = ["test", "hello", "world"]
    df2 = df.copy()
    df2.columns = ["pytest", "hello", "world"]
    return [df, df2]

def test_rename_first_col(dfs_to_test):
    pd.testing.assert_frame_equal(rename_first_col(dfs_to_test[0],'pytest'),dfs_to_test[1])
    pd.testing.assert_series_equal(rename_first_col(dfs_to_test[0], 'pytest').iloc[:,0], dfs_to_test[1].iloc[:,0])

def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    """
    Check that weighted_average raise zero division error when the sum of the weights is 0
    """
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([1,2],[-1,1])

