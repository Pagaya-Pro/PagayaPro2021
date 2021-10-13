import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def means_sampling(s, n):
    """
    Samples sqrt(len(s)) samples of size n and calculates mean for each one
    :param s: Pandas Series of numerical values
    :param n: int, size of each sample
    :return: ndarray of shape (len(s),)
    """
    num_means = int(np.sqrt(len(s)))
    return np.mean(np.random.choice(s.values, size=(num_means, n)), axis=1)


def means_sampling_plotting(s, n):
    """
    Samples sqrt(len(s)) samples of size n, calculates mean for each one and plots histogram of the means along with the
    mean of s
    :param s: Pandas Series
    :param n: int, size of each sample
    :return: None
    """
    mean_samples = np.mean(np.random.choice(s.values, size=(num_means, n)), axis=1)
    mean_of_means = mean_samples.mean()
    s_mean = s.values.mean()
    plt.title(f'Distribution of the Means - Series mean is {s_mean:.3f}, n = {n}')
    plt.xlabel('Mean Value')
    plt.axvline(x=mean_of_means, label='Mean of Means', color='r')
    sns.distplot(mean_samples, label='Means')
    plt.legend()
