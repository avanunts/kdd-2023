import numpy as np
import matplotlib.pyplot as plt


def get_session_length_dist(ds):
    prev_items = ds.prev_items
    x_counts, y_counts = np.unique(np.fromiter(map(len, prev_items), np.int32), return_counts=True)
    num_samples = ds.shape[0]
    y_cdf = [0] * (len(y_counts) + 1)
    x_cdf = [0] + x_counts.tolist()
    y_cdf[0] = 0
    for i in range(1, len(y_counts) + 1):
        y_cdf[i] = y_cdf[i - 1] + y_counts[i - 1] / num_samples
    return np.array(x_cdf), np.array(y_cdf)

def get_counts(l):
    counts = {}
    for elem in l:
        if elem not in counts:
            counts[elem] = 0



"""
s_len - full session length
prefix - prev_items
Returns frequency of prefix from ds 
"""


# def get_prefix_conditional_probs(ds, s_len):
#     prefix_length = ds.prev_items.apply(len)
#     prefixes = ds[prefix_length == s_len - 1].prev_items.tolist()
#

def plot_session_length_dists(x_max, **configs):
    x_cdfs, y_cdfs = {}, {}
    for name, config in configs.items():
        x_cdfs[name], y_cdfs[name] = get_session_length_dist(config['ds'])
    plt.figure(figsize=(16, 9))
    x_ticks, y_ticks = [0, x_max], [0.0, max([y_cdfs[name][x_max] for name in configs.keys()])]
    for name, config in configs.items():
        x_cdf, y_cdf = x_cdfs[name], y_cdfs[name]
        x_scatter = config['x_scatter']
        x_hlines = config['x_hlines']
        plt.plot(x_cdf[0:x_max], y_cdf[0:x_max], color=config['color'], label=name)
        plt.scatter(
            x_cdf[x_scatter],
            y_cdf[x_scatter],
            color=config['color']
        )
        plt.hlines(y_cdf[x_hlines], [0] * len(x_hlines), x_cdf[x_hlines], linestyles='dashed', color=config['color'])
        x_ticks = x_ticks + x_cdf[x_hlines].tolist()
        y_ticks = y_ticks + y_cdf[x_hlines].tolist()
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.ylim(bottom=0, top=1)
    plt.xlim(left=0)
    plt.legend()
    plt.show()
