def means_of_samples(S, n):
    """
        The function takes a pandas series S and a natural number n and creates a series of means of samples (with replacement) of n entries from S.

        Parameters:
            S (pandas series): data to be sampled.
            n (natural number): number of samples per mean.
        Returns:
            Pandas series: A series of means of samples.
    """
    import random
    random.seed(99)
    k = 1000
    means = []
    return S.sample(n * k, replace = True).values.reshape(k, n).mean(axis = 1)

def hist_means(S, n):
    """
    The function takes a pandas series S and a natural number n and plots a histogram of the series of means, and also indicates the mean of this series.

    Parameters:
        S (pandas series): data to be sampled.
        n (natural number): number of samples per mean.
    """
    sample = means_of_samples(S, n)
    import matplotlib.pyplot as plt
    plt.hist(sample)
    plt.xlabel('Activation probability')
    plt.ylabel('Counts')
    plt.title(f'Mean: {sample.mean():.3f}, Sample size: {n}')