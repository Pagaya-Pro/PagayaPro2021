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


#test this function, make sure for example please_test_me("testing is great") = "testing is great!!!"
def please_test_me(string: str) -> str:
    return string + "!!!"

#answer here
def test_please_test_me():
    assert please_test_me("testing is great") == 'testing is great!!!'

def times_7(number: Union[int, float]):
    return number * 7

#responses with parametrizing and fixtures are below
def test_make_me_2_functions_one_use_fixture_and_one_use_parametrize():
    assert times_7(2) == 14
    assert times_7(4) == 28
    assert times_7(0) == 0
    assert times_7(-1) == -7
    assert np.isclose(times_7(-0.1),-0.7) #added_the_extra_assertion_here

    random_generator = random.Random()
    for i in range(10):
        rnd_int = random_generator.randint(-1000, 1000)
        # time_7(rnd_int) is like summing 7 items of rnd_int
        assert times_7(rnd_int) == sum([rnd_int for i in range(7)])
        #assert times_7(rnd_int) > rnd_int  #Explain why this assert doest work. # response below
        #response: This doesn't work because the statement is only true for positive numbers. It fails on 0 and negatives.

#make_me_2_functions_one_use_fixture_and_one_use_parametrize
#using paramterizing
@pytest.mark.parametrize("value_to_multiply, product7", [(2, 14), (4, 28), (0, 0), (-1, -7), (-0.1, -0.7)])
def test_times_7_using_parametrize(value_to_multiply, product7):
    assert np.isclose(times_7(value_to_multiply),product7)

#using fixture
# I would be happy for feedback on this and whether this is the kind of thing
# you were looking for. I decided to split it to have one function that generates the numbers
# (something we might want to use in the future) and another function that tests them for
# assertions with times_7.
@pytest.fixture
def random_generator(num_random_ints=10,min=-1000,max=1000):
    random_generator = random.Random()
    rand_ints = []
    for i in range(num_random_ints):
        rand_ints.append(random_generator.randint(min, max))
    return rand_ints

#is it possible to do this with parametrize, using the output of the fixture above?
def test_times_7_using_fixture(random_generator):
    for i in random_generator:
        assert times_7(i) == sum([i for j in range(7)])



#Add a function and at least 3 tests
#DiagnosticCalculatorUsingBayesTheorem

def bayes_medical_diag(sensitivity,specificity,prevalence, diag_type=0):
    if diag_type == 0: #What is the probability someone with a positive result has the disease?
        return (sensitivity*prevalence/
                (sensitivity*prevalence+(1-specificity)*(1-prevalence)))
    elif diag_type == 1: #What is the probability someone with a negative result does not have the disease?
        return ((1-sensitivity)*prevalence/
                ((1-sensitivity)*prevalence+(specificity)*(1-prevalence)))
    else:
        return 'Invalid input. Type must be 0 or 1'

#test1&2
@pytest.mark.parametrize("sensitivity, specificity, prevalence, diag_type, result", [(0.99,0.98,0.005,0,0.1991),(0.99,0.98,0.005,1,0.00005)])
def test_bayes_medical_diag_parametrize(sensitivity, specificity, prevalence, diag_type, result):
    assert np.isclose(bayes_medical_diag(sensitivity,specificity,prevalence, diag_type),result,atol=1e-02)

#test3
def test_bayes_medical_diag():
    assert type(bayes_medical_diag(0.99,0.98,0.005,2)) == str

#add a function that get data frame as an argument and return it after some preprocess/change
#test the function you wrote use assert_frame_equal and assert_series_equal

@pytest.fixture
def hp_dataframe():
    data = dict(
        house=['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin', 'Gryffindor', 'Gryffindor', 'Ravenclaw'],
        scores=[402, 450, 234, 351, 702, 234, 300],
        year=[1993, np.nan, 1993, 1993, 1995, 1989, np.nan]
    )
    return pd.DataFrame(data)

def fillna_groupby_house_first(df):
    df.fillna(df.iloc[:, 2].median(), inplace=True)
    df[df.columns[2]] = df[df.columns[2]].astype('int')
    return df.groupby(df.columns[0]).first()

#assertframe
def test_fillna_groupby_house_first(hp_dataframe):
    """
    tests frame on fill_na and displaying first entry of each house after groupby
    """
    tested_df = fillna_groupby_house_first(hp_dataframe)
    expected_df = pd.DataFrame(dict(house = ['Gryffindor','Hufflepuff','Ravenclaw','Slytherin'],
                                    scores = [402,234,450,351],
                                    year = [1993,1993,1993,1993])
                               ).set_index('house')
    pd.testing.assert_frame_equal(tested_df,expected_df)

#assertseries
def test_fillna_groupby_house_first(hp_dataframe):
    """
    tests house series after fill_na and displaying first entry of each house after groupby
    """
    tested_df = fillna_groupby_house_first(hp_dataframe)
    expected_series = pd.Series([1993,1993,1993,1993],name = 'year', index = ['Gryffindor','Hufflepuff','Ravenclaw','Slytherin'])
    expected_series.index.names = ['house']
    pd.testing.assert_series_equal(tested_df['year'],expected_series)


def compute_weighted_average(x: List[float], w: List[float]) -> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)


def test_weighted_average_raise_zero_division_error():
    """
    tests for raising a ZeroDivisionError
    """
    with pytest.raises(ZeroDivisionError):
        assert compute_weighted_average([1,2,3],[-1,-2,3])  #check that weighted_average raise zero division error when the sum of the weights is 0
