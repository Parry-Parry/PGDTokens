import pyterrier as pt 
if not pt.started(): pt.init()
from pyterrier.io import read_results
import logging, argparse, os, time
import pandas as pd
from tqdm import tqdm
from fire import Fire
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import ir_datasets as irds 
from . import SemanticHelper, basicConfig, BERTWordRecover
from .attacker import Attacker
from utility import yaml_load
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)

model = None 
tokenizer = None
sem_help = None
word_re = None
name = None
attacker = Attacker()
TOPK = 100
MAX_ITER = 3

def optimise(query_idx,
             docs):
    word_idx = docs[0]
    attacker.get_model_gradient(model, query_idx, docs)
    attacker.attack(model, query_idx, docs, word_idx, name, max_iter=MAX_ITER)

def get_words(self, embeddings : torch.Tensor):
    attacked_matrix = word_re.get_word_embedding(model)
    attacked_matrix = attacked_matrix.detach().cpu().numpy()

    sim = np.dot(attacked_matrix, embeddings.T)
    sim_order = np.argsort(-sim, axis=0)[:, 1:1 + TOPK]
    # use word_re to get words
    words = [word_re.idx2word[idx] for idx in sim_order]
    return words

def main(config : str):
    config = basicConfig(**yaml_load(config))
    model = BertModel(config.model_id)
    tokenizer = BertTokenizer.from_pretrained(config.model_id)
    for _name, _ in model.named_parameters():
        name = _name
    dataset = irds.load(config.dataset)
    documents = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(dataset.query_iter()).set_index('query_id').text.to_dict()
    docs = read_results(config.data_path)
    embeddings = np.load(config.embeddings)
    # group by query id and make list of documents
    ranking_lookup = docs.groupby('qid')['docno'].apply(list).to_dict()
    initial_score_lookup = docs.set_index(['qid', 'docno'])['score'].to_dict() 

    for row in ranking_lookup.iter_tuples():
        qid = row.qid
        docnos = row.docno
        target_docs = [documents[docno] for docno in docnos]
        original_scores = [initial_score_lookup[(qid, docno)] for docno in docnos]  
        ranking = {docno : (tokenizer.encode(text).input_ids.to_list(), score) for docno, text, score in zip(docnos, target_docs, original_scores)}
        target_query = tokenizer.encode(queries[qid]).input_ids.to_list()
        
        for target_docno, (text, _) in ranking.items():
            current = ranking.copy()
            current.pop(target_docno)
            other_docs = [t for k, (t,s) in current.items()]
            docs = [text] + other_docs
            optimise(target_query, docs)
    
    new_candidates = get_words(embeddings)
    with open(config.out_dir, 'w') as f:
        for cand in new_candidates:
            f.write(cand + '\n')

if __name__ == "__main__":
    Fire(main)