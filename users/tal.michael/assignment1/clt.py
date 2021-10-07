import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def CLT(S, n):
    num_samples = 1000
    arr = np.random.choice(S, size = (num_samples, n), replace = True)
    return pd.Series(arr.mean(axis = 1))

def plot_CLT(S, n):
    ser = CLT(S, n)
    title = 'CLT histogram of {} entries \n mean: {:.3f} percent'.format(n, ser.mean())
    sns.distplot(ser, bins = int(np.sqrt(ser.unique().shape)), color = "lightpink")
    plt.title(title)
    plt.grid(True)
    plt.show()