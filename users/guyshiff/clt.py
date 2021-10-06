import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sample(S, n):
    """
    Creates means of random samples
    :param S:
    :param n:
    :return:
    """
    length = 1000
    S = S.to_numpy()
    indices = np.random.randint(S.size, size=(length, n))
    return S[indices].mean(axis=1)


def histogram_samples(S, n):
    """
    Plots histograms of samples distribution
    :param S:
    :param n:
    :return:
    """
    samples = sample(S, n)
    bins_num = max(int(np.sqrt(np.unique(samples).size)), 10)
    plt.hist(samples, bins=bins)
    plt.figure()