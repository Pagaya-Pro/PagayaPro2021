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


# test this function, make sure for example please_test_me("testing is great") = "testing is great!!!"
def please_test_me(string: str) -> str:
    return string + "!!!"


@pytest.mark.parametrize("input, output", [("testing is great", "testing is great!!!")])
def test_please_test_me_using_parametrize(input, output):
    assert please_test_me(input) == output


def times_7(number: Union[int, float]):
    return number * 7


@pytest.mark.parametrize("input, output", [(2, 14), (4, 28), (0, 0), (-1, -7), (1.5, 10.5), ('s', 'sssssss')])
def test_times_7_using_parametrize(input, output):
    assert times_7(input) == output


@pytest.fixture
def first_times_7():
    """
    This fixture is the first elements of the fibonacci series.
    (In real life this better be a constant, we use fixture for generating objects we need for testing)
    """
    products = []
    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        products.append((rnd_int, rnd_int * 7))
    return products


def test_times_7_with_fixture(first_times_7):
    for productValue in first_times_7:
        assert times_7(productValue[0]) == productValue[1]


# def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
#     assert times_7(2) == 14
#     assert times_7(4) == 28
#     assert times_7(0) == 0
#     assert times_7(-1) == -7
#     #  add one interesting case I didn't check
#
#     random_generator = random.Random()
#     for i in range(10):
#         rnd_int = random_generator.randint(-1000, 1000)
#         # time_7(rnd_int) is like summing 7 items of rnd_int
#         assert times_7(rnd_int) == sum([rnd_int for i in range(7)])

# assert times_7(rnd_int) > rnd_int  #Won't work for non-positive values. "-7<-1"

# Add a function and at least 3 tests
def factorial_of_none_negative(num):
    if (int(num) != num) or (num < 0):
        raise ArithmeticError("Not a non-negative integer")
    if num <= 1:
        return 1
    num_arr = np.arange(1, num + 1)
    return np.prod(num_arr)


@pytest.fixture
def first_factorials():
    return [1, 1, 2, 6, 24, 120, 720, 5040, 40320]


def test_factorial_of_none_negative(first_factorials):
    for index_value, productValue in enumerate(first_factorials):
        assert factorial_of_none_negative(index_value) == productValue


@pytest.mark.parametrize("item_index, factorial_value", [(0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (5, 120)])
def test_factorial_of_none_negative(item_index, factorial_value):
    assert factorial_of_none_negative(item_index) == factorial_value


@pytest.mark.parametrize("faulty_input", [-1, 4.676])
def test_exception_of_factorial_of_none_negative(faulty_input):
    with pytest.raises(ArithmeticError):
        assert factorial_of_none_negative(faulty_input)


# add a function that get data frame as an argument and return it after some preprocess/change
def df_to_lowercase(df):
    """
    Returns a copy of the dataframe where all of its string columns are in lowercase
    """
    data_lower_case = df.copy()
    for col in data_lower_case.columns:
        try:
            data_lower_case[col] = data_lower_case[col].str.lower()
        except:
            continue
    return data_lower_case


# test the function you wrote use assert_frame_equal and assert_series_equal
def test_df_to_lowercase_assert_frame_equal():
    df = pd.DataFrame({'name': ['Guy', 'nOa', 'daN', 'KOBI', 'IRis', 'lIoRa'], 'age': [28, 33, 24, 60, 60, 84],
                       'gender': ['MALE', 'FEMALE', 'MALE', 'MALE', 'femALE', 'fEMALe']})
    df_expected = pd.DataFrame({'name': ['guy', 'noa', 'dan', 'kobi', 'iris', 'liora'], 'age': [28, 33, 24, 60, 60, 84],
                                'gender': ['male', 'female', 'male', 'male', 'female', 'female']})
    pd.testing.assert_frame_equal(df_to_lowercase(df),df_expected)


def test_df_to_lowercase_assert_series_equal():
    df = pd.DataFrame({'name': ['Guy', 'nOa', 'daN', 'KOBI', 'IRis', 'lIoRa'], 'age': [28, 33, 24, 60, 60, 84],
                       'gender': ['MALE', 'FEMALE', 'MALE', 'MALE', 'femALE', 'fEMALe']})
    df_expected = pd.DataFrame({'name': ['guy', 'noa', 'dan', 'kobi', 'iris', 'liora'], 'age': [28, 33, 24, 60, 60, 84],
                                'gender': ['male', 'female', 'male', 'male', 'female', 'female']})
    data_lowered_case = df_to_lowercase(df)
    for col in data_lowered_case.columns:
        pd.testing.assert_series_equal(data_lowered_case[col],df_expected[col])


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)

# check that weighted_average raise zero division error when the sum of the weights is 0
def test_weighted_average_raise_zero_division_error():
    weighted_to_zero_sum = [-4,-3,-2,-1,0,1,2,3,4]
    x_values = [1,2,3,4,5,6,7,8,9]
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average(x_values,weighted_to_zero_sum)


