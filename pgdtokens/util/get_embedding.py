import numpy as np 
import torch
from transformers import BertTokenizer, BertModel
from os.path import join

def embed_vocab(self, model_id : str, out_dir : str):
    tok = BertTokenizer.from_pretrained(model_id)
    vocab = tok.get_vocab()

    emb = BertModel.from_pretrained(model_id)

    lookup = []

    for k, _ in vocab.items():
        ids = tok.encode(k).input_ids
        embedding = emb(torch.tensor(ids)).last_hidden_state.mean(dim=1).detach().numpy()
        lookup.append(embedding)
    
    lookup = np.array(lookup)

    # dot product with self
    sim = np.dot(lookup, lookup.T)

    np.save(join(out_dir, 'vocab.np'), vocab)
    np.save(join(out_dir, 'sim.np'), sim)