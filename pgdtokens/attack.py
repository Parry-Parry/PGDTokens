import pyterrier as pt 
if not pt.started(): pt.init()
from pyterrier.io import read_results
import logging
import pandas as pd
from tqdm import tqdm
from fire import Fire
import torch
from transformers import AutoTokenizer, AutoModel
import ir_datasets as irds 
from . import basicConfig, BERTWordRecover, SemanticHelper
from .attacker import Attacker
from utility import load_yaml
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)

attacker = Attacker()
TOPK = 100
MAX_ITER = 3

def optimise(query_idx,
             docs,
             model,
             param_name):
    word_idx = docs['input_ids'][0]
    attacker.get_model_gradient(model, query_idx, docs)
    attacker.attack(model, query_idx, docs, word_idx, param_name, max_iter=MAX_ITER)

def get_words(model,
              word_re : BERTWordRecover,
              sem_helper : SemanticHelper):
    attacked_matrix = word_re.get_word_embedding(model)
    attacked_matrix = attacked_matrix.detach().cpu().numpy()

    sim = np.dot(attacked_matrix, sem_helper.embeddings.T)
    sim_order = np.argsort(-sim, axis=0)[:, 1:1 + TOPK]
    # use word_re to get words
    words = [sem_helper.idx2word[sem_helper.embed2id[idx]] for idx in sim_order]
    return words

def main(config : str):
    config = basicConfig(**load_yaml(config))
    model_id = config.model_id
    dataset = config.dataset
    data_path = config.data_path
    out_dir = config.out_dir

    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sem_helper = SemanticHelper(config.embedding_path, tokenizer)
    sem_helper.build_vocab()

    for _name, _ in model.named_parameters():
        param_name = _name

    word_re = BERTWordRecover(param_name, tokenizer)

    dataset = irds.load(dataset)
    documents = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id').text.to_dict()
    docs = read_results(data_path)
    # group by query id and make list of documents
    ranking_lookup = docs.groupby('qid')['docno'].apply(list).to_dict()
    initial_score_lookup = docs.set_index(['qid', 'docno'])['score'].to_dict() 

    for row in tqdm(ranking_lookup.iter_tuples()):
        qid = row.qid
        docnos = row.docno
        target_docs = [documents[docno] for docno in docnos]
        original_scores = [initial_score_lookup[(qid, docno)] for docno in docnos]  
        ranking = {docno : (text, score) for docno, text, score in zip(docnos, target_docs, original_scores)}
        target_query = tokenizer.encode(queries[qid]).input_ids.to_list()
        
        for target_docno, (text, _) in ranking.items():
            current = ranking.copy()
            current.pop(target_docno)
            other_docs = [t for k, (t,s) in current.items()]
            docs = [text] + other_docs
            input_ids = tokenizer(docs, return_tensors='pt', padding=True).input_ids
            optimise(target_query, input_ids, model, param_name)
    
    new_candidates = get_words(model, word_re=word_re, sem_helper=sem_helper)
    with open(out_dir, 'w') as f:
        for cand in new_candidates:
            f.write(cand + '\n')

if __name__ == "__main__":
    Fire(main)