
def calculate_mean_on_n_samples(ser, n):
    NUM_OF_ELEM_IN_SAMPLE = 1000
    '''
    :param ser: a series
    :param n: number of samples
    :return: series of means of the samples
    '''
    return pd.Series([ser.sample(n=int(n), replace=True).mean() for i in range(NUM_OF_ELEM_IN_SAMPLE)])

def plot_histogram_clt(ser,n):
    '''
    :param ser: a series
    :param n: number of samples
    :return: prints the mean and plots a histogram of the n sample means.
    '''
    series_for_hist = calculate_mean_on_n_samples(ser,n)
    series_for_hist.hist(bins=30);
    print(f'The mean is: {series_for_hist.mean()}')
