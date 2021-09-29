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
    assert please_test_me("testing is great") == "testing is great!!!"


def times_7(number: Union[int, float]):
    return number * 7


# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
@pytest.fixture
def numbers_7():
    return [2, 4, 0, -1, 0.5]


@pytest.fixture
def numbers_times_7():
    return [14, 28, 0, -7, 3.5]


def test_times_7_fixture(numbers_7, numbers_times_7):
    for index, number in enumerate(numbers_7):
        assert times_7(number) == numbers_times_7[index]


@pytest.mark.parametrize("numbers, numbers_times_7", [(2, 14), (4, 28), (0, 0), (-1, -7), (0.5, 3.5)])
def test_times_7_parametrize(numbers, numbers_times_7):
    assert times_7(numbers) == numbers_times_7


def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
    assert times_7(2) == 14
    assert times_7(4) == 28
    assert times_7(0) == 0
    assert times_7(-1) == -7
    # TODO add one interesting case I didn't check
    assert times_7(0.5) == 3.5

    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])

        # assert times_7(rnd_int) > rnd_int  # TODO Explain why this assert doest work
        # A negative number multiplied by 7 is smaller and not larger


# TODO Add a function and at least 3 tests
def module_2(number):
    return number % 2


def test_module_2():
    assert module_2(2) == 0
    assert module_2(1) == 1
    assert module_2(0) == 0


@pytest.fixture
def numbers_2():
    return [2, 1, 0]


@pytest.fixture
def numbers_module_2():
    return [0, 1, 0]


def test_module_2_fixture(numbers_2, numbers_module_2):
    for index, number in enumerate(numbers_2):
        assert module_2(number) == numbers_module_2[index]


@pytest.mark.parametrize("numbers, numbers_module_2", [(2, 0), (1, 1), (0, 0)])
def test_module_2_parametrize(numbers, numbers_module_2):
    assert module_2(numbers) == numbers_module_2


# TODO add a function that get data frame as an argument and return it after some preprocess/change
def df_drop_na(df):
    return df.copy().dropna().reset_index(drop=True)


# TODO test the function you wrote use assert_frame_equal and assert_series_equal
def test_df_drop_na():
    pd.testing.assert_frame_equal(df_drop_na(pd.DataFrame(data=[1, 2, 3, 4, 5, float('nan'), 7.5])),
                                  pd.DataFrame(data=[1, 2, 3, 4, 5, 7.5]))
    pd.testing.assert_series_equal(df_drop_na(pd.Series(data=[1, 2, 3, 4, 5, float('nan'), 7.5])),
                                   pd.Series(data=[1, 2, 3, 4, 5, 7.5]))


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    """
    Test the compute_weighted_average function
    """
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([1, 2, 3], [0, 1, -1])
