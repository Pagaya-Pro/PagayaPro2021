import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



def random_sample_mean(S, n):
    n_means = 1000
    res = []
    for idx in range(n_means):
        res.append(S.sample(n, replace=True).mean())
    return pd.Series(res)



def random_sample_mean2(S, n):
    n_means = 10000
    # res = []
    # for idx in range(n_means):
    # res.append(S.sample(n, replace=True).mean())
    res = (np.random.choice(S, size=(n_means, n), replace=True).mean(axis=1))
    # res = pd.Series(res)
    plt.hist(res, bins=100, alpha=0.5)
    plt.title("Distribution of sample means,  " +str(n_means ) +" means,  " +str(n ) +" samples per mean", size=16)
    print("The mean of means is: " ,res.mean())

    return res


