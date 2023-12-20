import json
import numpy as np 
import torch
from transformers import AutoTokenizer, AutoModel
from os.path import join
from fire import Fire
from tqdm import tqdm

def embed_vocab(model_id : str, out_dir : str, sim : bool = False):
    tok = AutoTokenizer.from_pretrained(model_id)
    vocab = tok.get_vocab()

    emb = AutoModel.from_pretrained(model_id)

    lookup = {}
    for k, _ in tqdm(vocab.items()):
        emb_input = tok.encode(k, add_special_tokens=False, return_tensors='pt')
        embedding = emb(emb_input).last_hidden_state[0, :].detach().numpy()
        assert embedding.shape == (768,), embedding.shape
        lookup[k] = embedding.to_list()
    
    with open(join(out_dir, 'vocab.json'), 'w') as f:
        json.dump(lookup, f)
    # dot product with self
    if sim:
        sim = np.dot(lookup, lookup.T)
        np.save(join(out_dir, 'sim.npy'), sim)

if __name__ == "__main__":
    Fire(embed_vocab)