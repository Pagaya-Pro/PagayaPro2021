import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def create_mean_samples_and_plot_fast(values,n,num_of_means=2500):
    max_ind = len(values)-1
    num_of_bins = int(np.sqrt(num_of_means))
    random_sample = np.random.randint(0,max_ind,n*num_of_means)
    result = np.array(values[random_sample]).reshape(n,num_of_means)
    mean_list = result.mean(axis=0)
    panda_series = pd.Series(mean_list)
    return panda_series,result

def hist_CLT(values,n,num_of_means=2500):
    series,result = create_mean_samples_and_plot_fast(values,n)
    series.hist(bins=int(math.sqrt(series.size)))
    plt.axvline(result.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.title("number of samples "+str(n))
    plt.show()