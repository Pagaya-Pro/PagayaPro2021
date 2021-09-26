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

@pytest.mark.parametrize("str", ['Maor','Akav','ttt'])
def test_please_test_me(str):
    assert please_test_me(str) == f"{str}!!!"

def times_7(number: Union[int, float]):
    return number * 7


# TODO make_me_2_functions_one_use_fixture_and_one_use_parametrize
@pytest.mark.parametrize("num", [2,4,0,-1,-3])
def test_make_me_2_functions_and_one_use_parametrize(num):
    assert times_7(num) == (num * 7)

@pytest.fixture
def my_fixture():
    return 7

def test_make_me_2_functions_one_use_fixture(my_fixture):
    assert times_7(my_fixture) == (my_fixture * 7)

def test_make_me_2_functions():
    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])

        # Since we get negative random number and the 7 time is less then the number
        #assert times_7(rnd_int) > rnd_int  #





# TODO Add a function and at least 3 tests
@pytest.fixture
def matrix_result_example():
    return np.array([[ 5,  8, 14, 13],[ 8, 13, 23, 22],[11, 18 ,32, 31]])

@pytest.fixture
def matrix_1_example():
    return np.array([[1,2],[2,3],[3,4]])

@pytest.fixture
def matrix_2_example():
    return np.array([[1,2,4,5],[2,3,5,4]])

@pytest.fixture
def matrix_3_example():
    return np.array([[1,2,4],[2,5,4]])

@pytest.fixture
def matrix_inv_example():
    return np.array([[1,2,4],[2,3,4],[2,3,5]])

def math_mul(matrix1,matrix2):
    return matrix1.dot(matrix2)

def test_math_mul1(matrix_1_example,matrix_2_example,matrix_result_example):
    assert math_mul(matrix_1_example,matrix_2_example).all() == matrix_result_example.all()

def test_math_mul2(matrix_inv_example):
    assert math_mul(matrix_inv_example,np.linalg.inv(matrix_inv_example)).all() == np.identity(3).all()

def test_math_mul3(matrix_3_example,matrix_inv_example):
    assert math_mul(math_mul(matrix_3_example,matrix_inv_example),np.linalg.inv(matrix_inv_example)).all() == matrix_3_example.all()

def test_math_mul4(matrix_3_example):
    with pytest.raises(ValueError):
        math_mul(matrix_3_example,matrix_3_example)

# TODO add a function that get data frame as an argument and return it after some preprocess/change
# TODO test the function you wrote use assert_frame_equal and assert_series_equal


@pytest.fixture
def example_data_frame():
    return pd.DataFrame({"Name":["Maor","Amit"],"Age":[33,22]})

@pytest.fixture
def example_change_data_frame():
    return pd.DataFrame({"Name":["Maor","Amit"],"Age":[34,23]})

def change_data_frame(df):
    df["Age"] = df["Age"] + 1
    return df

def test_change_data_frame(example_data_frame,example_change_data_frame):
    pd.testing.assert_frame_equal(change_data_frame(example_data_frame), example_change_data_frame)

def test_change_series(example_data_frame, example_change_data_frame):
    pd.testing.assert_series_equal(change_data_frame(example_data_frame)["Age"], example_change_data_frame["Age"])


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)

# First check the function
def test_weighted_average():
    assert compute_weighted_average([1,2,1],[1,2,2]) == 1.4

def test_weighted_average_raise_zero_division_error():
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([1,2,3],[1,2,-3])