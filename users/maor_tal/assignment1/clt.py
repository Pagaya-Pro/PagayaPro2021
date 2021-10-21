import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# this function takes 500 sample of size sample_size and displays their means in a series.
def sample_means_generation(S,sample_size):
    sample_means = []
    for i in range(500):
        sample_indices = np.random.randint(len(S),size=sample_size)
        sample = S.iloc[sample_indices]
        sample_mean = sample.mean()
        sample_means.append(sample_mean)
    return pd.Series(sample_means, name='sample_mean')

#this function sketches the histogram of the distribution of the sample means
# and returns the mean of the sample mean distribution as a percentage out of 100

def histogram_of_sample_means(S,sample_size):
    sample_means = sample_means_generation(S,sample_size)
    mean_of_sample_means = sample_means.mean()
    sample_means.hist()
    plt.show()
    return mean_of_sample_means*100
