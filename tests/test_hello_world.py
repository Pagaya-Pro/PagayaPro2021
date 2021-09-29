"""
Examples of tests using pytest.
Read these functions and then fix the TODOs in the end of the file.
You can learn about pytest here:
https://www.guru99.com/pytest-tutorial.html
"""
import math
import random
from typing import Union, List
import pandas as pd
import numpy as np
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
    assert please_test_me("I love Pycharm") == "I love Pycharm!!!"
    assert please_test_me("this will work?") == "this will work?!!!"


def times_7(number: Union[int, float]):
    return number * 7

# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize

@pytest.mark.parametrize("num", [2,100,-1,-3,0, 5.3, 0.00000000001, -3.8])
def test_times_7_parametrize(num):
    assert times_7(num) == 7*num
    assert times_7(num) / 7 == num

@pytest.fixture
def random_arr(request):
    random_generator = random.Random()
    ret_arr = []

    for i in range(int(request.param)):
        ret_arr.append(random_generator.randint(-1000, 1000))
    return ret_arr

@pytest.mark.parametrize("random_arr", [10], indirect=True)
def test_times_7_fixture(random_arr):
    for rnd_num in random_arr:
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_num) == sum([rnd_num for i in range(7)])

        # assert times_7(rnd_int) > rnd_int  # TODO Explain why this assert doest work
        # This assert won't work since rnd_int can be negative thus rnd_int*7 < rnd_int

# TODO Add a function and at least 3 tests
def dist(x: tuple, y: tuple) -> float:
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

@pytest.mark.parametrize("x, y", [((1,1), (0,0)), ((2,5), (0,0)), ((-8,3), (0,0))])
def test_dist(x, y):
    assert round(dist(x,y), 4) == round(math.dist(x,y), 4)
    assert dist(x, y) == dist(y, x)
    assert dist(x, x) == 0

# TODO add a function that get data frame as an argument and return it after some preprocess/change
def change_df(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy.iloc[:,2] = 5
    return df_copy

# TODO test the function you wrote use assert_frame_equal and assert_series_equal
@pytest.mark.parametrize("random_arr", [100], indirect=True)
def test_change_df(random_arr):
    data = np.array(random_arr).reshape((25, 4))
    rnd_data_frame = pd.DataFrame(data=data, columns=['col_1', 'col_2', 'col_3', 'col_4'])

    pd.testing.assert_frame_equal(change_df(rnd_data_frame)[['col_1','col_2']],
                                            rnd_data_frame[['col_1','col_2']])
    pd.testing.assert_series_equal(change_df(rnd_data_frame)['col_4'],
                                             rnd_data_frame['col_4'])


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)

@pytest.mark.parametrize("random_arr", [50, 100], indirect=True)
def test_weighted_average_raise_zero_division_error(random_arr):
    # TODO check that weighted_average raise zero division error when the sum of the weights is 0
    with pytest.raises(ZeroDivisionError):
        ret = random_arr
        ret.append(-sum(ret))
        assert compute_weighted_average(random_arr, ret)