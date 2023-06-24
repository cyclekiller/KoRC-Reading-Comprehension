import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").cuda()
print('Passage encoding model loaded.')

def calc_emb(input):
    with torch.no_grad():
        input = tokenizer(input, padding=True, truncation=True, max_length=256, return_tensors='pt').to('cuda')
        batch_output = model(**input)
        emb =  mean_pooling(batch_output, input['attention_mask'])
        emb = F.normalize(emb, p=2, dim=1)
    return emb

if __name__ == '__main__':
    query = r'''Waterloo was the original name for the city of Austin , Texas , located in Travis in the central part of the state .After [] Vice President Mirabeau B. Lamar visited the area during a buffalo - hunting expedition between 1837 and 1838 , he proposed that the republic 's capital , then located in Houston , be relocated to an area situated on the north bank of the Colorado River near the present - day Ann W. Richards Congress Avenue Bridge in what is now central Austin .In 1839 , the site was officially chosen as the seventh and final location for the capital of the [Country A] .It was incorporated under the name " Waterloo " .Shortly thereafter , the name was changed to Austin in honor of Stephen F. Austin , the " Father of Texas " and the republic 's first secretary of state .'''
    query_emb = calc_emb(query)
    
    #Compute dot score between query and all document embeddings
    doc_emb = torch.from_numpy(np.load('step1_retrieve_doc/doc_embed.npy')).cuda()

    scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

    #Combine docs & scores
    from utils.doc import *
    doc_score_pairs = list(zip(docs, scores))

    #Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    #Output passages & scores
    for doc, score in doc_score_pairs[:3]:
        print(score, doc)
