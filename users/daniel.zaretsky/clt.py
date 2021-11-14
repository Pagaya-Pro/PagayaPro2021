import random
import matplotlib.pyplot as plt
def means_of_samples(S: pd.Series, n: int) -> pd.Series:
    """
        The function takes a pandas series S and a natural number n and creates a series of means of samples (with replacement) of n entries from S.
        Parameters:
            S (pandas series): data to be sampled.
            n (natural number): number of samples per mean.
        Returns:
            Pandas series: A series of means of samples.
    """
    k = 1000
    return S.sample(n * k, replace = True, random_state=99).values.reshape(k, n).mean(axis = 1)
def hist_means(S: pd.Series, n: int):
    """
    The function takes a pandas series S and a natural number n and plots a histogram of the series of means, and also indicates the mean of this series.
    Parameters:
        S (pandas series): data to be sampled.
        n (natural number): number of samples per mean.
    """
    means_series = means_of_samples(S, n)
    plt.hist(means_series)
    plt.xlabel('Activation probability')
    plt.ylabel('Counts')
    plt.title(f'Mean: {means_series.mean():.3f}, Sample size: {n}')
