def random_sample_mean(s, n):
    return s[np.random.randint(low=0, high=len(s) - 1, size=n)].mean()


def bootstrap_samples(s: pd.Series, n: int):
    ret_size = int(len(s) ** 0.5)

    temp = np.ndarray(shape=(ret_size, len(s)))
    temp[np.arange(ret_size)] = s

    return np.apply_along_axis(random_sample_mean, axis=1, arr=temp, n=n)

def hist_boot(s, n):
    tmp = bootstrap_samples(s, n)
    plt.hist(tmp)
    return tmp.mean()

hist_boot(np.random.randint(low=0, high=1000000-1, size=1000000), 1000000)