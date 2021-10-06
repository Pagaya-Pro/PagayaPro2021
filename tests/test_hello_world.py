"""
Examples of tests using pytest.
Read these functions and then fix the TODOs in the end of the file.
You can learn about pytest here:
https://www.guru99.com/pytest-tutorial.html
"""
import random
import pandas as pd
from typing import Union
from typing import List

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


@pytest.fixture
def some_random_strings():
    """
    This fixture is the first elements of the fibonacci series.
    (In real life this better be a constant, we use fixture for generating objects we need for testing)
    """
    return ['aaa','!!!']


def test_please_test_me__using_fixture(some_random_strings):
    assert please_test_me(some_random_strings[0]) == "aaa!!!"
    assert please_test_me(some_random_strings[1]) == "!!!!!!"

'''
#@pytest.mark.parametrize("item_index, fibonacci_value", [(0, 1), (1, 1), (2, 2), (4, 5), (6, 13), (8, 34)])
#def test_fibonacci_using_parametrize(item_index, fibonacci_value):
@pytest.mark.parametrize("strings","output", [("aaa", "aaa!!!"), ("!!!", "!!!!!!")])
def test_please_test_me__using_parametrise(strings,output):
    print(strings)
    assert please_test_me(strings) == output
'''
@pytest.mark.parametrize("strings, output", [("aaa", "aaa!!!"), ("!!!", "!!!!!!")])
def test_please_test_me_using_parametrise(strings, output):
    print(strings)
    assert please_test_me(strings) == output

def times_7(number: Union[int, float]):
    return number * 7


# DONE make_me_2_functions_one_use_fixture_and_one_use_parametrize see please_test_me
def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
    assert times_7(2) == 14
    assert times_7(4) == 28
    assert times_7(0) == 0
    assert times_7(-1) == -7
    assert times_7([0]) == [0,0,0,0,0,0,0]
    # DONE add one interesting case I didn't check

    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])

        # assert times_7(rnd_int) > rnd_int  # TODO Explain why this assert doest work: because negative numbers would break this test


# DONE (in the most tierd way ever) Add a function and at least 3 tests
def is_the_pizza_expensive(pizza): #-> df
    pizza['expensive'] = pizza['price'] > 30

def test_is_the_pizza_expensive():
    pizza_dict = {'price':[20,40]}
    pizza = pd.DataFrame(data = pizza_dict)
    is_the_pizza_expensive(pizza)
    ans_dict = {'expensive': [False, True]}
    ans = pd.DataFrame(data = ans_dict)
    assert(len(pizza.columns) == 2)
    pd.testing.assert_frame_equal(pizza[['expensive']], ans)
    pd.testing.assert_series_equal(pizza.expensive, ans.expensive)

# DONE add a function that get data frame as an argument and return it after some preprocess/change
# DONE test the function you wrote use assert_frame_equal and assert_series_equal


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    with pytest.raises(ZeroDivisionError):
        compute_weighted_average(x=[1.0,-1.0,0.0], w=[1.0, -1.0, 0])# DONE check that weighted_average raise zero division error when the sum of the weights is 0