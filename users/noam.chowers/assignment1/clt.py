import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def sample_mean(S, n, iterations=100, seed=123):
    """
    :param S: A pandas series for mean distribution calculations
    :param n: Number of samples in each iteration
    :param iterations: Number of means to calculate
    :param seed: for recreation
    :return: pandas series of means
    """
    # Set seed
    random.seed(seed)
    mean_arr = []
    # Iterate and append a mean based on n samples
    for i in range(iterations):
        mean_arr.append(S.sample(n=n, replace=True).mean())
    return pd.Series(mean_arr)

def plot_means(S, n, ax, iterations=10000, seed=123):
    """
    :param S, n, iterations, seed: as the sample_mean function
    :param ax: the plot axes
    :return: None
    """
    # Grab the mean array
    mean_arr = sample_mean(S, n, iterations, seed)
    # Calculate it's mean
    mean_of_means = mean_arr.mean()
    # Histogram Density Plot
    sns.distplot(mean_arr, ax=ax)
    # Add a mean line in red
    ax.axvline(mean_of_means, color='r')
    # Configure visualization
    ax.set_xlabel('Mean')
    ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title('n={0}'.format(n))

# This is how we call the functions above.
"""
n_vals = np.array([1, 2, 5, 10, 50, 100, 1000, 10**6])

fig, ax = plt.subplots(2, 4,
                      constrained_layout=True)

fig.set_size_inches(18.5, 10.5, forward=True)

for i, n in enumerate(n_vals):
    j=0
    if i >= 4:
        j = 1
        i -= 4
    plot_means(S, n, ax[j][i])

plt.suptitle('Mean Histogram', fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""