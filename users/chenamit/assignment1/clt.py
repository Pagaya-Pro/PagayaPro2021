import pandas as pd
import numpy as np

def sample(S, n):
    means = []
    for i in range(int(np.sqrt(len(S)))):
        samp = S.sample(n=n, replace=True)
        mean = np.array(samp).mean()
        means.append(mean)
    return means
