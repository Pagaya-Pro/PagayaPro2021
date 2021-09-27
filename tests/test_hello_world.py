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


def please_test_me(string: str) -> str:
    return string + "!!!"


@pytest.mark.parametrize("input_sentence, output_sentence", [("testing is great", "testing is great!!!")])
def test_please_test_me(input_sentence, output_sentence):
    assert please_test_me(input_sentence) == output_sentence


def times_7(number: Union[int, float]):
    return number * 7


@pytest.mark.parametrize("input, output", [(2, 14),(4,28),(0,0),(-1,-7),(0.5,3.5)])
def test_times_7_parametrize(input, output):
    assert times_7(input) == output


@pytest.fixture
def first_multiplies_of_7():
    """
    This fixture is the first 10 no negative multiplication of 7.
    """
    random_generator = random.Random()
    result = []
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        result.append((rnd_int,7*rnd_int))

    # Add test for non integer input
    result.append((0.5,3.5))
    return result


def test_times_7_using_fixture(first_multiplies_of_7):
    """
    Test the times_7 function. Tests the first elements of the series.
    Args:
        first_multiplies_of_7: This is a fixture so it is automatically filled.
            The first_multiplies_of_7 will have the first 10 no negative multiplication of 7
            see first_multiplies_of_7() function.
    """
    for test_tupple in first_multiplies_of_7:
        assert times_7(test_tupple[0]) == test_tupple[1]


def calc_factorial(n: int):
    if not ((n >= 0) and (int(n) == n)):
        raise ArithmeticError("cant calculate factorial of non integer:",n)
    elif (n <= 1):
        return 1

    return np.prod(np.arange(1,n+1))


@pytest.mark.parametrize("input, output", [(0,1),(1,1),(2,2),(4,24),(6,720)])
def test_calc_factorial(input, output):
    assert calc_factorial(input) == output


@pytest.fixture
def first_positive_integers_factorials():
    """
    This fixture is the first 6 positive integers and their respective factorials
    """
    return [1,1,2,6,24,120,720]


def test_factorial_fixture(first_positive_integers_factorials):
    """
    Test the factorial function. Tests the first positive integers.
    Args:
        first_positive_integers_factorials: This is a fixture so it is automatically filled.
            The first_positive_integers_factorials will have the first 6 positive integers respective factorials
    """
    for item_index, factorial in enumerate(first_positive_integers_factorials):
        assert calc_factorial(item_index) == factorial


@pytest.mark.parametrize("bad_input", [0.5,-1])
def test_factorial_exception(bad_input):
    """
    Test the factorial function. Tests that floating point number raises an arithmetic error.
    """
    with pytest.raises(ArithmeticError):
        assert calc_factorial(bad_input)


def round_floats_in_df(df):
    """
    return a copy of the df with all the numeric columns rounded to the closes integer.
    """
    for column in df.columns:
        try:
            df[column] = np.round(df[column])
        except:
            continue
    return df


def test_round_floats_in_df():
    """
    Test the round_floats_in_df function. Tests an example data frame with 2 columns.
    Use @assert_frame_equal to verify the output matches the expected result
    """
    df = pd.DataFrame({"name" : ["Yuval","Dani","Yossi","Ron"],"Exam_Grade" : [89.5,85.7,87,94.2],"HW_Grade" : [97.2,89.3,91.7,92.1]})
    expected_df = pd.DataFrame({"name" : ["Yuval","Dani","Yossi","Ron"],"Exam_Grade" : [90.0,86.0,87.0,94.0],"HW_Grade" : [97.0,89.0,92.0,92.0]})
    pd.testing.assert_frame_equal(round_floats_in_df(df),expected_df)


def test_round_floats_in_df_series():
    """
    Test the round_floats_in_df function. Tests an example data frame with 2 columns.
    Use @assert_series_equal to verify that each output column matches it's parallel column in the expected result dataframe
    """
    df = pd.DataFrame({"name" : ["Yuval","Dani","Yossi","Ron"],"Exam_Grade" : [89.5,85.7,87,94.2],"HW_Grade" : [97.2,89.3,91.7,92.1]})
    expected_df = pd.DataFrame({"name" : ["Yuval","Dani","Yossi","Ron"],"Exam_Grade" : [90.0,86.0,87.0,94.0],"HW_Grade" : [97.0,89.0,92.0,92.0]})
    round_floats_in_df(df)
    for column in df.columns:
        pd.testing.assert_series_equal(df[column],expected_df[column])


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    """
    Test the compute_weighted_average function. Tests an example where the sum of weight is zero.
    Use @with to verify that compute_weighted_average returns ZeroDivisionError in case of zero sum weights
    """
    zero_sum_weights = [1,2,-2,-3,3,-1]
    x_values = [1,2,3,4,5,6]
    print(sum(zero_sum_weights))
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average(x_values,zero_sum_weights)
