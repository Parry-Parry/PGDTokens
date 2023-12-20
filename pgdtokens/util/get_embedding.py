import numpy as np 
import torch
from transformers import BertTokenizer, BertModel
from os.path import join
from fire import Fire

def embed_vocab(model_id : str, out_dir : str, sim : bool = False):
    tok = BertTokenizer.from_pretrained(model_id)
    vocab = tok.get_vocab()

    emb = BertModel.from_pretrained(model_id)

    lookup = []
    for k, _ in vocab.items():
        ids = tok.encode(k)
        embedding = emb(**torch.tensor(ids)).last_hidden_state.mean(dim=0).detach().numpy()
        lookup.append(embedding)
    
    lookup = np.array(lookup)
    np.save(join(out_dir, 'vocab.npy'), vocab)

    # dot product with self
    if sim:
        sim = np.dot(lookup, lookup.T)
        np.save(join(out_dir, 'sim.npy'), sim)

if __name__ == "__main__":
    Fire(embed_vocab)