import gc
import torch
from step1_retrieve_doc.embed_util import calc_emb
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils.doc import *
doc_to_entity = None

class MyDocData(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

dataset = MyDocData(docs)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

if __name__ == '__main__':
    all_embeddings = []

    # Batch processing on GPU
    with torch.no_grad():
        for batch in tqdm(dataloader):
            embeddings = calc_emb(batch).cpu().numpy()
            all_embeddings.append(embeddings)

    tokenizer = None
    model = None
    dataset = None
    dataloader = None
    docs = None
    gc.collect()

    # Concatenate the outputs from all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(all_embeddings.shape)

    np.save(open('step1_retrieve_doc/doc_embed.npy', 'wb'), all_embeddings, allow_pickle=False)
