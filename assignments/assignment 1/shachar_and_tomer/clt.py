def CLT(s, n):
    m = int(len(s)/2)
    samples = s.sample(n*m, replace = True).to_numpy().reshape(n,m)
    samples_mean = samples.mean(axis = 0)
    return samples_mean

def plotHist(s, n):
    x = CLT(s, n)
    ax = sns.kdeplot(x, shade=False, color='crimson')
    kdeline = ax.lines[0]
    mean = x.mean()
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    height = np.interp(mean, xs, ys)
    ax.vlines(mean, 0, height, color='crimson', ls=':')
    ax.fill_between(xs, 0, ys, facecolor='crimson', alpha=0.2)
    plt.xlabel("Activation rate")
    return plt