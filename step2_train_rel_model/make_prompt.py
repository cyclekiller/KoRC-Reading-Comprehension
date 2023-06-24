import numpy as np
import json
from tqdm import tqdm
from utils.doc import *

for data_split in ['train', 'valid']:
    data = json.load(open(f"data/{data_split}.json", 'r'))
    doc_ids = np.load(f"step1_retrieve_doc/{data_split}_retrieved_docs.npy")
    
    all_questions = []
    error = 0
    error2 = 0
    for i  in tqdm(range(len(data))):
        oracle = docs[doc_ids[i]]
        passage = data[i]['passage']
        try:
            remaining = passage.split('[', 1)[1]
            unknown = remaining.split(']', 1)[0] # passage contains some empty []s
        except IndexError:
            error += 1
            all_questions += ['Error parsing the text\nError parsing the text\nError parsing the text']
            continue
        while unknown == '':
            try:
                remaining = remaining.split('[', 1)[1]
            except IndexError:
                error2 += 1
                break
            unknown = remaining.split(']', 1)[0]
        my_question = oracle + '\n' + passage + '\n' + f'What is the name of this [{unknown}]?'
        all_questions += [my_question]
    
    print(error) # train: 36, valid: 20
    print(error2) # train: 17, valid: 3
    with open(f'step3_resolve_unknown_entity/{data_split}_prompt.txt', 'w') as f:
        f.write('\n\n'.join(all_questions))