import pandas as pd
import numpy as np
import argparse
import os
from utils import plot_cdf
import math
import json
from gensim.corpora import Dictionary

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, 
        help='input file in csv format with at least 4 columns: "srcip", "dstip", "ts", "td"')
parser.add_argument('--corpus_dir', type=str, 
        help='directory for preprocessed corpus', default="corpus")
parser.add_argument('--mfc', type=int, help='minimum flow count', default=200)
parser.add_argument('--nbfs', type=int, help='number of bins for flow size', default=1e6)
parser.add_argument('--nbit', type=int, help='number of bins for interarrival time', default=1e6)
parser.add_argument('--logscale', action='store_true', help='use log scale for cutting flow size and interarrival time into bins')
args = parser.parse_args()

# read data
raw_data = pd.read_csv(args.input_file)

# discard flows with zero flow size
raw_data = raw_data[raw_data['td'] > 0]

# discard src-dst pairs with flow count < MIN_FLOW_COUNT
sd_group = raw_data.groupby(['srcip', 'dstip'])
sd_group = sd_group.filter(lambda x: len(x) >= args.mfc).groupby(['srcip', 'dstip'])

# count overall distribution of flow size and interarrival time
total_fs_list = raw_data['td'] # flow size
total_it_list = [] # interarrival time
for name, group in sd_group:
    total_it_list.extend(list(np.diff(group['ts'])))

# plot cdf of flow size and interarrival time for raw data
hyper_str = f"mfc-{args.mfc}_nbfs-{args.nbfs}_nbit-{args.nbit}"
os.makedirs(os.path.join(args.corpus_dir, hyper_str), exist_ok=True)

plot_cdf(total_it_list, "Interarrival Time", "CDF", 
        os.path.join(args.corpus_dir, hyper_str, "cdf_it.pdf"), 
        "Real", x_logscale=False, y_logscale=False)
plot_cdf(total_it_list, "Interarrival Time", "CDF", 
        os.path.join(args.corpus_dir, hyper_str, "cdf_it_logscale.pdf"), 
        "Real", x_logscale=True, y_logscale=True)
plot_cdf(total_fs_list, "Flow Size", "CDF", 
        os.path.join(args.corpus_dir, hyper_str, "cdf_fs.pdf"), 
        "Real", x_logscale=False, y_logscale=False)
plot_cdf(total_fs_list, "Flow Size", "CDF", 
        os.path.join(args.corpus_dir, hyper_str, "cdf_fs_logscale.pdf"), 
        "Real", x_logscale=True, y_logscale=True)

min_fs, max_fs = min(total_fs_list), max(total_fs_list)
min_it, max_it = min(total_it_list), max(total_it_list)

min_nonzero_it = min([x for x in total_it_list if x > 0])
# num_zero_it = total_it_list.count(0)
percentage_zero_it = total_it_list.count(0) / len(total_it_list)

EPSILON = 1e-6
if not args.logscale:
    # linear binning
    bin_size_fs = (max_fs - min_fs) / args.nbfs
    bin_size_it = (max_it - min_it) / args.nbit
else:
    # log binning
    bin_size_fs = (math.log(max_fs + EPSILON) - math.log(min_fs + EPSILON)) / args.nbfs
    bin_size_it = (math.log(max_it + EPSILON) - math.log(min_it + EPSILON)) / args.nbit

def get_bin_idx(val, bin_size, min_val, log_scale=False):
    if not log_scale:
        bin_idx = math.floor((val - min_val) / bin_size)
    else:
        bin_idx = math.floor((math.log(val + EPSILON) - math.log(min_val + EPSILON)) / bin_size)
    return bin_idx

def get_bin_idx_list(vals, bin_size, min_val, log_scale=False):
    bin_idx_list = []
    for val in vals:
        bin_idx = get_bin_idx(val, bin_size, min_val, log_scale)
        bin_idx_list.append(bin_idx)
    return bin_idx_list

corpus, src_dst_list, raw_flows = [], [], []
for name, group in sd_group:
    fs_list = group['td']
    it_list = [0] + list(np.diff(group['ts']))
    fs_bin_idx_list = get_bin_idx_list(fs_list, bin_size_fs, min_fs, args.logscale)
    it_bin_idx_list = get_bin_idx_list(it_list, bin_size_it, min_it, args.logscale)
    flow_idx_list = [str(fs_bin_idx) + "_" + str(it_bin_idx) for fs_bin_idx, 
                    it_bin_idx in zip(fs_bin_idx_list, it_bin_idx_list)]
    corpus.append(flow_idx_list)
    src_id, dst_id = name
    src_dst_list.append(f"{src_id}_{dst_id}")
    raw_flows.extend(group[["srcip", "dstip", "ts", "td"]].values.tolist())

# save preprocessed corpus
# bin_method = "log" if args.logscale else "linear"
os.makedirs(os.path.join(args.corpus_dir, hyper_str), exist_ok=True)
with open(os.path.join(args.corpus_dir, hyper_str, "corpus.json"), "w") as f:
    json.dump(corpus, f)
with open(os.path.join(args.corpus_dir, hyper_str, "src_dst_list.json"), "w") as f:
    json.dump(src_dst_list, f)
pd.DataFrame(raw_flows, columns=["srcip", "dstip", "ts", "td"]).to_csv(
    os.path.join(args.corpus_dir, hyper_str, "raw_flows.csv"), index=False)

# create and save dictionary
dictionary = Dictionary(corpus)
dictionary.save(os.path.join(args.corpus_dir, hyper_str, "dictionary.dict"))

# save hyperparameter configuration for training
with open("config-pre.json", "w") as f:
    config = vars(args)
    config["min_fs"] = min_fs
    config["min_it"] = min_it
    config["min_nonzero_it"] = min_nonzero_it
    config["percentage_zero_it"] = percentage_zero_it
    config["bin_size_fs"] = bin_size_fs
    config["bin_size_it"] = bin_size_it
    json.dump(config, f, indent=4)