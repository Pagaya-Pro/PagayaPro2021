import pandas as pd
import numpy as npgit

def clt(S, n):
    m = 1000
    indices = np.random.randint(len(S),size=(n,m))
    return pd.Series(S.values[indices].mean(axis=0))


def clt_with_plot(S, n):
    m = 1000
    indices = np.random.randint(len(S),size=(n,m))
    res = pd.Series(S.values[indices].mean(axis=0))
    avg = res.mean()
    sns.distplot(res).set(title='The mean is {}'.format(avg))
    return res.mean()