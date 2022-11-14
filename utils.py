import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 15})
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# color-blindness friendly
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def vals2cdf(vals):
    dist_dict = dict(Counter(vals))
    dist_dict = {k: v for k, v in sorted(dist_dict.items(), key = lambda x: x[0])}
    x = dist_dict.keys()

    pdf = np.asarray(list(dist_dict.values()), dtype=float) / float(sum(dist_dict.values()))
    cdf = np.cumsum(pdf)
    return x, cdf


def plot_cdf(vals, xlabel, ylabel, plot_loc, label,  x_logscale=False, y_logscale=False):
    plt.clf()
    x, cdf = vals2cdf(vals)
    plt.plot(x, cdf, label=label, color=CB_color_cycle[0], linewidth=5)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if x_logscale:
        plt.xscale('log')
    if y_logscale:
        plt.yscale('log')

    plt.savefig(plot_loc, bbox_inches="tight", dpi=300)


# syn_vals_dict: {name: vals}
def plot_cdf2(raw_vals, syn_vals_dict, xlabel, ylabel, plot_loc, x_logscale=False, y_logscale=False):
    plt.clf()
    x, cdf = vals2cdf(raw_vals)
    plt.plot(x, cdf, label="Real", color=CB_color_cycle[0], linewidth=5)
    idx = 1
    for method, syn_vals in syn_vals_dict.items():
        x, cdf = vals2cdf(syn_vals)
        plt.plot(x, cdf, label=method, color=CB_color_cycle[idx], linewidth=1.5)
        idx += 1
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if x_logscale:
        plt.xscale('log')
    if y_logscale:
        plt.yscale('log')
    plt.savefig(plot_loc, bbox_inches="tight", dpi=300)