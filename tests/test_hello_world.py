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


def test_str_ret_val():
    assert please_test_me("testing is great") == "testing is great!!!"


def times_7(number: Union[int, float]):
    return number * 7


# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
    assert times_7(2) == 14
    assert times_7(4) == 28
    assert times_7(0) == 0
    assert times_7(-1) == -7
    # TODO add one interesting case I didn't check
    assert times_7(1 / 7) == 1

    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])

        assert times_7(rnd_int) > rnd_int  # TODO Explain why this assert doest work
                                           # This assert doesn't work because the random number is negative, and
                                           # multiplication with positive number makes it smaller


@pytest.mark.parametrize("test_num, test_val", [(2, 14), (4, 28), (0, 0), (-1, -7), (1 / 7, 1)])
def test_with_parametrize(test_num, test_val):
    assert times_7(test_num) == test_val


@pytest.fixture
def get_random_numbers():
    random_generator = random.Random()
    return [random_generator.randint(-1000, 1000) for _ in range(10)]


def test_with_fixture(get_random_numbers):
    for rnd_int in get_random_numbers:
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])


# TODO Add a function and at least 3 tests

def make_pos(matrix):
    return np.abs(matrix)

@pytest.fixture()
def neg_matrices():
    return [-1 * np.random.random((3, 3)) for _ in range(5)]

@pytest.fixture()
def pos_matrices():
    return [np.random.random((3, 3)) for _ in range(5)]

@pytest.fixture()
def mix_matrices():
    return [np.random.randn(3, 3) for _ in range(5)]


def test_make_pos_with_neg(neg_matrices):
    for mat in neg_matrices:
        assert np.all(make_pos(mat) > mat)
        assert np.sum(make_pos(mat)) > np.sum(mat)


def test_make_pos_with_pos(pos_matrices):
    for mat in pos_matrices:
        assert np.all(make_pos(mat) == mat)
        assert np.sum(make_pos(mat)) == np.sum(mat)


def test_make_pos_with_mix(mix_matrices):
    for mat in mix_matrices:
        assert np.all(make_pos(mat) >= mat)
        assert np.sum(make_pos(mat)) >= np.sum(mat)


# TODO add a function that get data frame as an argument and return it after some preprocess/change
def rotate_cols(df):
    new_cols = [df.columns.to_list()[-1]] + df.columns.to_list()[:-1]
    return df[new_cols]

@pytest.fixture()
def test_dfs():
    random_data = np.random.random((20, 5))
    rotated_data = np.concatenate((np.expand_dims(random_data[:, -1], -1), random_data[:, :-1]), axis=1)
    reg_df = pd.DataFrame(data=random_data, columns=['a', 'b', 'c', 'd', 'e'])
    rotated_df = pd.DataFrame(data=rotated_data, columns=['e', 'a', 'b', 'c', 'd'])
    return reg_df, rotated_df


# TODO test the function you wrote use assert_frame_equal and assert_series_equal
def test_rotate_cols(test_dfs):
    df, rotated_df = test_dfs
    to_test = rotate_cols(df)
    pd.testing.assert_frame_equal(rotated_df, to_test)
    pd.testing.assert_series_equal(df['e'], to_test.iloc[:, 0])


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    pass  # TODO check that weighted_average raise zero division error when the sum of the weights is 0
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([1, 2], [-1, 1])