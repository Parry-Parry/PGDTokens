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

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)


def attack_probing_bertdot(model, 
                           query,
                           docs,
                           word_re,
                           attack_idx,
                           batch_list, docid_list, attack_doc_id, tokenizer, sem_help, word_re, ori_score, ori_qds, qid):
    
    attacker = Attacker()

    matrix = word_re.get_word_embedding(model)
    frozen_matrix = matrix.clone().detach()
    attacker.get_model_gradient(model, query, docs)
    candidate = word_re.idx2word[attack_idx]

    attacker.attack()



def attack_by_testing_blackbox(ori_model, surr_model, batch_list, docid_list,
                               attack_doc_id, args,
                               bert_tokenizer, semantic_helper, word_re,
                               ori_score, ori_qds, qid):

    attack_doc_input, batch_list = find_attack_doc_input_and_remove(batch_list,
                                                                    docid_list,
                                                                    attack_doc_id)

    attack_input_ids_list = attack_doc_input['input_ids'].tolist()
    sep_token_id = bert_tokenizer.sep_token_id

    query_token_id = attack_input_ids_list[1:attack_input_ids_list.index(
                                           sep_token_id)]

    with_last_sep_doc_token_ids_list = attack_input_ids_list[
                                       attack_input_ids_list.index(
                                           sep_token_id) + 1:]
    ori_doc_token_ids_list = with_last_sep_doc_token_ids_list[
                             :len(with_last_sep_doc_token_ids_list) - 1]

    doc_token_ids_list = list({}.fromkeys(ori_doc_token_ids_list).keys())

    word_embedding_matrix = word_re.get_word_embedding(surr_model)
    ori_we_matrix = word_embedding_matrix.clone().detach()

    attacker = Attacker()

    attacker.get_model_gradient(surr_model, batch_list, attack_doc_input,
                                args.device)

    word_idx = 0
    for word_idx in gradient_topk_word_idx_list:
        gradient_topk_words.append(word_re.idx2word[word_idx])

    attacker.attack(surr_model, batch_list, attack_doc_input,
                    attack_word_idx=doc_token_ids_list,
                    args=args, eps=args.eps, max_iter=args.max_iter)

    attacked_we_matrix = word_re.get_word_embedding(surr_model)

    sim_word_ids_dict, sim_values, sub_word_dict = semantic_helper.pick_most_similar_words_batch(
        gradient_topk_words, ori_doc_token_ids_list, word_re,
        args.simi_candi_topk, args.simi_threshod)

    new_doc_token_id_list, score = \
        word_re.recover_document_greedy_rank_pos(ori_doc_token_ids_list, ori_we_matrix,
                                                 attacked_we_matrix, sim_word_ids_dict, ori_score,
                                                 ori_model, query_token_id, args,
                                                 sub_word_dict, ori_qds, attack_doc_id, qid)

    return new_doc_token_id_list, score


def main(config : dict):
    config = basicConfig(config)

    model = BertModel(config.model_id)
    tokenizer = BertTokenizer.from_pretrained(config.model_id)
    out_dir = config.out_dir
    attack_id = config.attack_id
    init_token = tokenizer.encode(config.init_token).input_ids[1]

    dataset = irds.load(config.dataset)
    documents = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(dataset.query_iter()).set_index('query_id').text.to_dict()
    docs = read_results(config.data_path)
    # group by query id and make list of documents
    ranking_lookup = docs.groupby('qid')['docno'].apply(list).to_dict()
    initial_score_lookup = docs.set_index(['qid', 'docno'])['score'].to_dict() 

    for row in ranking_lookup.iter_tuples():
        qid = row.qid
        docnos = row.docno
        target_docs = [documents[docno] for docno in docnos]
        original_scores = [initial_score_lookup[(qid, docno)] for docno in docnos]  
        ranking = {docno : (tokenizer.encode(text).input_ids.to_list(), score) for docno, text, score in zip(docnos, target_docs, original_scores)}
        target_query = queries[qid]
        
        for docno, (text, score) in ranking.items():
            current = ranking.copy()
            current.pop(docno)
            text = [init_token] + text[1:] if attack_id == 0 else text[1:] + [init_token]

if __name__ == "__main__":
    Fire(main)