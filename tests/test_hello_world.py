"""
Examples of tests using pytest.
Read these functions and then fix the TODOs in the end of the file.
You can learn about pytest here:
https://www.guru99.com/pytest-tutorial.html
"""
import random
from typing import Union
from typing import List
from functools import reduce
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
    assert please_test_me("testing is great") == "testing is great!!!"


def times_7(number: Union[int, float]):
    return number * 7


# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
@pytest.mark.parametrize("multiply, expected_result", [(2, 14), (4, 28), (0, 0), (-1, -7), (0.5, 3.5)])
def test_make_me_2_functions_one_use_parametrize(multiply, expected_result):
    assert times_7(multiply) == expected_result

        # assert times_7(rnd_int) > rnd_int  # TODO Explain why this assert doest work
        # The above assertion doesn't work because for negative values, multiplying by 7 isn't greater than the original number


@pytest.fixture
def time_7_numbers():
    random_generator = random.Random()
    test_vals = []
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        test_vals.append((rnd_int, sum([rnd_int for j in range(7)])))
    return test_vals


def test_make_me_2_functions_one_use_fixture(time_7_numbers):
    for num_pair in time_7_numbers:
        assert times_7(num_pair[0]) == num_pair[1]


# TODO Add a function and at least 3 tests
def my_factorial(n: int):
    if n < 0:
        raise ValueError("Value must be zero or positive")
    f = 1
    for i in range(1,n+1):
        f *= i
    return f


@pytest.mark.parametrize("number", [0, 1, 4])
def test_my_factorial(number):
    from math import factorial
    assert my_factorial(number) == factorial(number)


def test_my_factorial2():
    with pytest.raises(ValueError):
        assert my_factorial(-1)


def test_my_factorial3():
    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(1, 1000)
        assert my_factorial(rnd_int) == reduce(lambda x, y: x * y, [i for i in range(1, rnd_int+1)])


# TODO add a function that get data frame as an argument and return it after some preprocess/change
def sum_group_by_first_col(df):
    return df.groupby(df.columns[0]).sum()


# TODO test the function you wrote use assert_frame_equal and assert_series_equal
@pytest.fixture
def test_df():
    return pd.DataFrame({'Team': [0, 1, 1, 2, 1, 3, 2], 'earned': [1, 1, 2, 1, 1, 2, 1]})


def test_sum_group_by_first_col(test_df):
    expected_values = pd.DataFrame({'Team': [0, 1, 2, 3], 'earned': [1, 4, 2, 2]}).set_index('Team')
    expected_series = pd.Series({0: 1, 1: 4, 2: 2, 3: 2}, name='earned')
    pd.testing.assert_frame_equal(sum_group_by_first_col(test_df), expected_values)
    pd.testing.assert_series_equal(sum_group_by_first_col(test_df).reset_index(drop=True).squeeze(), expected_series)


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average(x=[1, 2, 3, 4], w=[1, 2, -3])
