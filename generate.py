from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np
import json
import argparse
import os
import math
from tqdm import tqdm
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, help='config file in json format')
parser.add_argument('--time_limit', type=int, help='time limit in seconds', default=1e8)
parser.add_argument('--output_dir', type=str, help='directory for output', default='output')
args = parser.parse_args()

# load config
with open(args.config_file, 'r') as f:
    config = json.load(f)
hyper_str = f"mfc-{config['mfc']}_nbfs-{config['nbfs']}_nbit-{config['nbit']}"
corpus_dir = config['corpus_dir']
model_dir = config['model_dir']
model_name = config['model_name']
min_fs = config['min_fs']
min_it = config['min_it']
bin_size_fs = config['bin_size_fs']
bin_size_it = config['bin_size_it']
min_nonzero_it = config['min_nonzero_it']
percentage_zero_it = config['percentage_zero_it']
logscale = config['logscale']

# load model
dictionary = Dictionary.load(os.path.join(corpus_dir, hyper_str, "dictionary.dict"))
model = LdaModel.load(os.path.join(model_dir, hyper_str, model_name, "model"))
doc_topics_matrix = np.load(os.path.join(model_dir, hyper_str, model_name, "doc_topics_matrix.npy"))
topic_terms_matrix = model.get_topics()

# load src_dst_list
with open(os.path.join(corpus_dir, hyper_str, "src_dst_list.json"), 'r') as f:
    src_dst_list = json.load(f)

# convert discrete flow size and interarrival time to continuous values
EPSILON = 1e-6
def sample(bin_idx, bin_size, min_val, log_scale=False):
    low = bin_idx * bin_size
    high = (bin_idx + 1) * bin_size
    val = np.random.uniform(low=low, high=high)
    if log_scale:
        return math.exp(val + math.log(min_val + EPSILON))
    else:
        return val + min_val


pair_wise_dir = os.path.join(args.output_dir, hyper_str, model_name, "pair_wise")
os.makedirs(pair_wise_dir, exist_ok=True)

# generate flows
print(f"total number of src_dst pairs: {len(src_dst_list)}")

syn_flows = []
for idx, src_dst in tqdm(enumerate(src_dst_list)):
    srcip, dstip = src_dst.split("_")
    ts = 0
    os.makedirs(os.path.join(pair_wise_dir, src_dst), exist_ok=True)
    pair_wise_flows = []
    while ts < args.time_limit:
        # sample topic
        theta = doc_topics_matrix[idx]
        topic_idx = np.argmax(np.random.multinomial(1, theta))
        # sample flow
        beta = topic_terms_matrix[topic_idx]
        beta /= (np.sum(beta) + EPSILON)
        # print(np.sum(beta))
        flow_idx = np.argmax(np.random.multinomial(1, beta))
        flow_str = dictionary[flow_idx] # "fs-bin-idx_it-bin-idx"
        fs_bin_idx, it_bin_idx = flow_str.split("_")
        fs_bin_idx = int(fs_bin_idx)
        it_bin_idx = int(it_bin_idx)
        # sample flow size and interarrival time
        flow_size = sample(fs_bin_idx, bin_size_fs, min_fs, log_scale=logscale)
        interarrival = sample(it_bin_idx, bin_size_it, min_it, log_scale=logscale)
        while interarrival < min_nonzero_it:
            if random.random() < percentage_zero_it / 100:
                break
            interarrival = sample(it_bin_idx, bin_size_it, min_it, log_scale=logscale)
        # update timestamp
        ts += interarrival
        # add flow
        syn_flows.append((srcip, dstip, ts, flow_size))
        pair_wise_flows.append((srcip, dstip, ts, flow_size))
    pd.DataFrame(pair_wise_flows, columns=["srcip", "dstip", "ts", "td"]).to_csv(os.path.join(pair_wise_dir, src_dst, "syn.csv"), index=False)

# save all flows
pd.DataFrame(syn_flows, columns=["srcip", "dstip", "ts", "td"]).to_csv(os.path.join(args.output_dir, hyper_str, model_name, "syn_flows.csv"), index=False)