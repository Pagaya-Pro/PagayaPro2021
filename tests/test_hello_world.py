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


def please_test_me(string: str) -> str:
    return string + "!!!"

@pytest.mark.parametrize("string", ["testing is great"])
def test_please_test_me(string):
    assert please_test_me(string) == "testing is great!!!"


def times_7(number: Union[int, float]):
    return number * 7


@pytest.mark.parametrize("num, sol", [(2, 14), (4, 28), (0, 0), (-1, -7)])
def test_times_7_using_param(num, sol):
    assert times_7(num) == sol

@pytest.fixture
def generate_numbers():
    random_generator = random.Random()
    return [random_generator.randint(-1000, 1000) for i in range(10)]

def test_times_7_using_fixture(generate_numbers):
    for i in range(len(generate_numbers)):
        assert times_7(generate_numbers[i]) == sum([generate_numbers[i] for j in range(7)])

        # assert times_7(rnd_int) > rnd_int  # TODO Explain why this assert doest work
        # this assert doesn't work because if rnd_int is a negative number it is not true.


def devide_by_two(num):
    return num/2

@pytest.mark.parametrize("num, sol", [(2, 1), (4.2, 2.1), (0, 0), (-1, -0.5)])
def test_devide_by_two(num, sol):
    assert devide_by_two(num) == sol



def change_df_index(df):
    df = df.set_index(pd.Index(range(1,df.shape[0]+1)))
    return df


def test_change_df_index():
    data = dict(person_name = ["Ron", "Roy", "Shay"], values = [3,6,2])
    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame(data, index=[1,2,3])
    pd.testing.assert_frame_equal(change_df_index(df1), df2)

def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average((1,2,3), (-1,0,1))