from tqdm import tqdm
import numpy as np
from utils.constants import DEFAULT_QUESTION_TYPE

def step1():
    ''' retrieve the most relevant documents '''
    import json
    from step1_retrieve_doc.retrieve_doc_faiss import calc_emb, index
    for dataset in ['iid_test', 'ood_test']:
        data = json.load(open(f"submit/{dataset}.json", 'r'))
        batch_size = 128
        num_batches = (len(data) - 1) // batch_size + 1
        tot_I = []
        for i in tqdm(range(num_batches)):
            batch_query = [item['passage'] for item in data[i * batch_size: (i + 1) * batch_size]]
            query_emb = calc_emb(batch_query).cpu().numpy()
            _, I = index.search(query_emb, 1)
            tot_I.extend(I)
        np.save(f'submit/{dataset}_retrieved_docs.npy', np.array(tot_I).squeeze(), allow_pickle=False)

def step2_1():
    ''' embed the question sentence using bert '''
    import torch
    from transformers import AutoTokenizer, AutoModel
    import pandas as pd

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to('cuda')
    
    for dataset in ['iid_test', 'ood_test']:
        df = pd.read_json(f"submit/{dataset}.json", orient='records')
        tokenized = tokenizer([item[DEFAULT_QUESTION_TYPE] for item in df["question"]], padding = True, truncation = True, return_tensors="pt")

        batch_size = 128

        tokenized = {k: torch.tensor(v) for k, v in tokenized.items()}

        num_batches = (tokenized['input_ids'].size(0) - 1) // batch_size + 1

        with torch.no_grad():
            x = torch.empty(0).to('cuda')

            for i in tqdm(range(num_batches)):
                batch_tokenized = {k: v[i * batch_size:(i + 1) * batch_size].to('cuda') for k, v in tokenized.items()}
                batch_hidden = model(**batch_tokenized).last_hidden_state[:, 0, :]
                x = torch.cat((x, batch_hidden), dim=0)

        x = x.to("cpu").numpy()
        np.save(open(f'submit/{dataset}_bert_feature.npy', 'wb'), x, allow_pickle=False)    

def step2_2():
    ''' predict relations between question and answer entity using trained classifier '''
    import torch
    from torch.utils.data import DataLoader
    from step2_train_rel_model.eval import my_model
    from utils.relation import idx2rel

    for dataset in ['iid_test', 'ood_test']:
        features = np.load(open(f'submit/{dataset}_bert_feature.npy', 'rb'))

        all_predicted = []
        data_loader = DataLoader(features, batch_size=64, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(data_loader):
                outputs = my_model(torch.tensor(batch).to('cuda'))
                _, predicted = torch.max(outputs, dim=1)
                all_predicted += predicted.cpu().numpy().tolist()
        
        with open(f'submit/{dataset}_predicted.txt', 'w') as f:
            for pred in all_predicted:
                print(idx2rel[pred], file=f)

def step3_1():
    ''' prepend the retrieved document before the passage'''
    import json
    from utils.doc import docs

    for data_split in ['iid_test', 'ood_test']:
        data = json.load(open(f"submit/{data_split}.json", 'r'))
        doc_ids = np.load(f"submit/{data_split}_retrieved_docs.npy")
        
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
        
        print(error)
        print(error2)
        with open(f'submit/{data_split}_prompt.txt', 'w') as f:
            f.write('\n\n'.join(all_questions))

def step3_2():
    ''' determine the possible span of question entity using dp algorithm'''
    import json
    from step3_resolve_unknown_entity.resolve_unknown_entity_dp import run

    for data_split in ['iid_test', 'ood_test']:
        with open(f'submit/{data_split}_prompt.txt', 'r') as f:
            all_questions = f.read().split('\n\n')
        data = json.load(open(f"submit/{data_split}.json", 'r'))
        assert(len(data) == len(all_questions))
        all_answers = []

        pbar = tqdm(range(len(all_questions)))
        for idx in pbar:
            try:
                x, y, _ = all_questions[idx].split('\n')
            except ValueError:
                all_answers += ['']
                continue
            entity = run(x, y)
            all_answers += [entity]
        with open(f'submit/{data_split}_answers_dp.txt', 'w') as f:
            f.write('\n'.join(all_answers))

def submit():
    ''' link question entity in the answer span, and use the predicted relation to make one-hop reasoning'''
    import json
    from utils.entity import entity2label
    from eval import get_entity_id, get_final_answer

    for data_split in ['iid', 'ood']:
        with open(f'submit/{data_split}_test_answers_dp.txt', 'r') as f:
            my_answers = f.readlines()
        data = json.load(open(f"submit/{data_split}_test.json", 'r'))
        my_rels = open(f'submit/{data_split}_test_predicted.txt', 'r').readlines()
        assert len(my_answers) == len(data) == len(my_rels)
        final_answers = {}
        for i in tqdm(range(len(my_answers))):
            entity_id = get_entity_id(my_answers[i]) # link the question entity
            final_answer = list(get_final_answer(entity_id, my_rels[i])) # one-hop reasoning
            if len(final_answer) > 100:
                final_answer = final_answer[:100]
            final_answer = [entity2label(item[1:]) for item in final_answer if item != '']
            final_answer = [item for item in final_answer if item != '<none>']
            final_answers[data[i]['id']] = final_answer
        json.dump(final_answers, open(f'submit/{data_split}.json', 'w'))

step1()
step2_1()
step2_2()
step3_1()
step3_2()
submit()