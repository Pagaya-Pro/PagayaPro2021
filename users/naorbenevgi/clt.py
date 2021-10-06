import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_mean_samples(series,n,num_of_means=10000):
    """
    Test the CLT theorem for a series of values. Randomly sample num_of_means groups of n values from a series element
    then return a list of the groups mean values.
    Args:
        ""series": the list of values being tested
        "n": the size of each sampled group.
        "num_of_means": number of groups to sample.
    """
    max_ind = len(series)-1
    return pd.Series([series[np.random.randint(0,max_ind,n)].mean() for i in range(num_of_means)])

def create_mean_samples_and_plot(series,n,num_of_means=10000):
    """
    Test the CLT theorem for a series of values and plot the results. The same as "create_mean_samples" with a plot.
    The mean of means will be marked on the resulting chart.
    Args:
        ""series": the list of values being tested
        "n": the size of each sampled group.
        "num_of_means": number of groups to sample.
    """
    max_ind = len(series)-1
    num_of_bins = int(np.sqrt(num_of_means))
    result = pd.Series([series[np.random.randint(0,max_ind,n)].mean() for i in range(num_of_means)])
    print("mean of the means of {} randomly sampled input groups is: {}".format(n,result.mean()))
    plt.hist(result, bins=num_of_bins, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(result.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.show()