import pandas as pd
import numpy as np
import argparse
import os
from utils import plot_cdf
import math
import json
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, help='config file in json format')
parser.add_argument('--num_topics', type=int, help='the number of topics in LDA', default=200)
parser.add_argument('--chunksize', type=int, help='chunksize in LDA', default=2000)
parser.add_argument('--passes', type=int, help='training passes in LDA', default=50)
parser.add_argument('--iterations', type=int, help='inference iterations in LDA', default=500)
parser.add_argument('--model_dir', type=str, help='dirctory for model', default='model')
args = parser.parse_args()

with open(args.config_file, 'r') as f:
    config = json.load(f)
hyper_str = f"mfc-{config['mfc']}_nbfs-{config['nbfs']}_nbit-{config['nbit']}"
corpus_dir = config['corpus_dir']

# load corpus
dictionary = Dictionary.load(os.path.join(corpus_dir, hyper_str, 'dictionary.dict'))
with open(os.path.join(corpus_dir, hyper_str, 'corpus.json'), 'r') as f:
    corpus = json.load(f)
corpus_bow = [dictionary.doc2bow(doc) for doc in corpus]

# train LDA model
_ = dictionary[0] # force to load dictionary
id2word = dictionary.id2token
model = LdaModel(
    corpus=corpus_bow,
    id2word=id2word,
    chunksize=args.chunksize,
    alpha='auto',
    eta='auto',
    iterations=args.iterations,
    num_topics=args.num_topics,
    passes=args.passes,
    eval_every=None
)

# save model
model_name = f"lda-{args.num_topics}_chunk-{args.chunksize}_pass-{args.passes}_iter-{args.iterations}"
os.makedirs(os.path.join(args.model_dir, hyper_str, model_name), exist_ok=True)
model.save(os.path.join(args.model_dir, hyper_str, model_name, "model"))

# transform doc_topics to a matrix of size (num_src_dst_pair, num_topics)
doc_topics = model.get_document_topics(corpus_bow)
doc_topics_matrix = np.zeros((len(doc_topics), args.num_topics))
for i, doc_topic in enumerate(doc_topics):
    for topic, prob in doc_topic:
        doc_topics_matrix[i][topic] = prob
    # normalize
    doc_topics_matrix[i] = doc_topics_matrix[i] / np.sum(doc_topics_matrix[i])

# save doc_topics_matrix
np.save(os.path.join(args.model_dir, hyper_str, model_name, "doc_topics_matrix.npy"), doc_topics_matrix)

# update and save config
config['model_name'] = model_name
config['model_dir'] = args.model_dir
with open("config-train.json", "w") as f:
    json.dump(config, f, indent=4)