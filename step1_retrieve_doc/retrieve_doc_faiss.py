import faiss                   # make faiss available
import numpy as np
import json
from tqdm import tqdm

d = 384

xb = np.load(open('step1_retrieve_doc/doc_embed.npy', 'rb'))

index = faiss.IndexFlatL2(d)   # build the index
# print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

from step1_retrieve_doc.embed_util import calc_emb
batch_size = 128

def retrieve_train_data():
    train_data = json.load(open("data/train.json", "r"))
    num_train_batches = (len(train_data) - 1) // batch_size + 1
    tot_I = []
    for i in tqdm(range(num_train_batches)):
        batch_query = [item['passage'] for item in train_data[i * batch_size: (i + 1) * batch_size]]
        query_emb = calc_emb(batch_query).cpu().numpy()
        _, I = index.search(query_emb, 1)
        tot_I.extend(I)
    np.save('step1_retrieve_doc/train_retrieved_docs.npy', np.array(tot_I).squeeze())

def retrieve_valid_data():
    val_data = json.load(open("data/valid.json", "r"))
    num_val_batches = (len(val_data) - 1) // batch_size + 1
    tot_I = []
    for i in tqdm(range(num_val_batches)):
        batch_query = [item['passage'] for item in val_data[i * batch_size: (i + 1) * batch_size]]
        query_emb = calc_emb(batch_query).cpu().numpy()
        _, I = index.search(query_emb, 1)
        tot_I.extend(I)
    np.save('step1_retrieve_doc/valid_retrieved_docs_1.npy', np.array(tot_I).squeeze())

if __name__ == '__main__':
    retrieve_train_data()
    retrieve_valid_data()