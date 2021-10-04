"""
Examples of tests using pytest.
Read these functions and then fix the TODOs in the end of the file.
You can learn about pytest here:
https://www.guru99.com/pytest-tutorial.html
"""
import random
from typing import Union, List
from faker import Faker
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
    assert please_test_me("testing_is_great") == "testing_is_great!!!"
    assert please_test_me('') == "!!!"

def times_7(number: Union[int, float]):
    return number * 7


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
        # FOR EXAMPLE: rnd_int=-1 -> times_7(-1) = -7 < -1 ->

@pytest.fixture
def multiplication_7():
    '''
    return a list of LIMIT multiplications of 7
    '''
    LIMIT = 10
    return [7*num for num in range(LIMIT)]

def test_times_7_fixture(multiplication_7):
    for i in range(len(multiplication_7)):
        assert times_7(i) == multiplication_7[i]

@pytest.mark.parametrize("num", [i for i in range(0,-10,-1)])
def test_times_7_parametrize(num):
    assert times_7(num) == (7 * num)


# TODO Add a function and at least 3 tests
def does_contain_my_name(name: str) -> bool:
    '''
    @ret == True iff name contains the string 'oded', lower or upper case
    '''
    name = str.lower(name)
    return 'oded' in name

def test_does_contain_my_name():
    assert does_contain_my_name("Ron Wettenstein") == False
    assert does_contain_my_name('oded goffer') == True
    assert does_contain_my_name('Oded Goffer') == True

@pytest.fixture
def generated_name():
    fake = Faker()
    return fake.name()

def test_does_contain_my_name_fixture(generated_name):
    name = str.lower(name_generator)
    assert does_contain_my_name(name_generator) == ('oded' in name)

@pytest.mark.parametrize("name", ['oded', 'yuval', 'OdeD', 'Guy', 'GOD'])
def test_does_contain_my_name_parametrize(name):
    assert (str.lower(name) == 'oded') == does_contain_my_name(name)


# TODO add a function that get data frame as an argument and return it after some preprocess/change
def num_frame_diagonal_to_trace(mat: pd.DataFrame):
    assert mat.shape[0] == mat.shape[1]
    trace = sum([mat.loc[i,i] for i in range(len(mat))])
    for i in range(len(mat)):
        mat.loc[i,i] = trace
    return mat

# TODO test the function you wrote use assert_frame_equal and assert_series_equal
def test_num_frame_diagonal_to_trace():
    mat = pd.DataFrame([[1,2],[3,4]])
    after_func = pd.DataFrame([[5,2],[3,5]])
    pd.testing.assert_frame_equal(num_frame_diagonal_to_trace(mat), after_func)

@pytest.mark.parametrize("series", [0,1])
def test_num_frame_diagonal_to_trace_series(series):
    mat = pd.DataFrame([[1,2],[3,4]])
    after_func = pd.DataFrame([[5,2],[3,5]])
    pd.testing.assert_series_equal(
        num_frame_diagonal_to_trace(mat)[series], after_func[series]
    )

def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)

def test_weighted_average_raise_zero_division_error():
    '''
    TODO check that weighted_average raise zero division error when the sum of the weights is 0
    '''
    with pytest.raises(ZeroDivisionError):
        compute_weighted_average([1,1,1],[-1,1,0])