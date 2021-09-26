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


@pytest.fixture
def ten_randoms_neg_1k_to_1k():
    random_generator = random.Random()
    rnd_ints = [random_generator.randint(-1000, 1000) for i in range(10)]
    return rnd_ints


def test_times_7_using_fixture():
    for rnd_int in ten_randoms_neg_1k_to_1k:
        assert times_7(rnd_int) == sum(rnd_int for i in range(7))


@pytest.mark.parametrize("number, number_multiplied_by_7", [(2, 14), (4, 28), (0, 0), (-1, -7), (0.5, 3.5)])
def test_times_7_using_parametrize(number, number_multiplied_by_7):
    assert times_7(number) == number_multiplied_by_7


# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
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
        # this does not hold in case rnd_int is negative.
        # also, function might return rnd_int+1 and still pass this assert


# TODO Add a function and at least 3 tests
def find_reciprocal(x: Union[int, float]):
    return 1/x


def test_find_reciprocal():
    assert find_reciprocal(1) == 1
    assert find_reciprocal(8) == 0.125
    assert find_reciprocal(0.2) == 5


# TODO add a function that get data frame as an argument and return it after some preprocess/change
def mess_with_df(df):
    first_col = df.columns[0]
    df[first_col] = 1
    return df


@pytest.fixture
def data_frame_example():
    df = pd.DataFrame({"first": [1, 2, 3, 4, 5], "B": [5, 5, 5, 5, 5]})
    return df


# TODO test the function you wrote use assert_frame_equal and assert_series_equal
def test_mess_with_df(data_frame_example):
    expected_result = pd.DataFrame({"first": [1, 1, 1, 1, 1], "B": [5, 5, 5, 5, 5]})
    pd.testing.assert_frame_equal(
        mess_with_df(data_frame_example), expected_result
    )
    pd.testing.assert_series_equal(
        mess_with_df(data_frame_example)["first"], expected_result["first"]
    )


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    # TODO check that weighted_average raise zero division error when the sum of the weights is 0
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([50, 12, 3], [1, 0, -1])
