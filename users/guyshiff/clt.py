import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_a_series_of_means_of_samples(S, n):
    """
    Creates means of random samples
    :param S: Series to sample from
    :param n: number of items in a sample
    :return: random samples means
    """
    length = 1000
    S = S.to_numpy()
    indices = np.random.randint(S.size, size=(length, n))
    return S[indices].mean(axis=1)


def histogram_samples(S, n):
    """
        Plots histograms of samples distribution
        :param S: Series to sample from
        :param n: number of items in a sample
        """
    samples = create_a_series_of_means_of_samples(S, n)
    bins_num = max(int(np.sqrt(np.unique(samples).size)), 10)
    plt.hist(samples, bins=bins_num)
    plt.figure()
