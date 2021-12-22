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
    assert please_test_me("testing is great") == "testing is great!!!"


def times_7(number: Union[int, float]):
    return number * 7


@pytest.mark.parametrize("a,results", [(2, 14), (4, 28), (0, 0), (-1, -7), (1, 7)])
def test_parametrize(a, results):
    assert times_7(a) == results

@pytest.fixture
def obj_fixture():
    return random.Random()

def test_fixture(obj_fixture):
    for i in range(10):
        rnd_int = obj_fixture.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])

        #assert times_7(rnd_int) > rnd_int  # Maybe rnd_int is a non positive number(zero or negative) and then
        # its not true.

# TODO Add a function and at least 3 tests
def add_to_shachar(string):
    return string + " thank you !"

def test_times_7_itay():
    add_str = " thank you !"
    str1 = "Ron is the king"
    str2 = "Ron can you help me with git?"
    str3 = "Ron what is the best question you had? "
    assert add_to_shachar(str1) == str1+add_str
    assert add_to_shachar(str2) == str2+add_str
    assert add_to_shachar(str3) == str3+add_str


# TODO add a function that get data frame as an argument and return it after some preprocess/change
def function_to_check(df):
    df["first_name"] = "Ron"
    return df

@pytest.fixture
def create_df():
    d = {'first_name': ["Itay","Ron"], 'last_name': ["Tshuva","Wettenstein"]}
    return pd.DataFrame(data=d)

#TODO test the function you wrote use assert_frame_equal and assert_series_equal
def test_function_to_check(create_df):
    cpy_df = create_df.copy()
    cpy_df["first_name"] = "Ron"
    tested_df = function_to_check(create_df)
    pd.testing.assert_frame_equal(cpy_df, tested_df)
    pd.testing.assert_series_equal(cpy_df["first_name"], tested_df["first_name"])
#


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)

# TODO check that weighted_average raise zero division error when the sum of the weights is 0
def test_weighted_average_raise_zero_division_error():
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([1,1],[1,-1])