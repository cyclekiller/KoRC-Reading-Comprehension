from utils.parse_data import *
from utils.entity2rel_dict import *
from utils.relation import *
from utils.constants import *
from tqdm import tqdm

lookup_dict = get_lookup_dict()

def make(original_data, split: str):
    '''
        count multiple answers as multiple (duplicated) questions;
        dedup_id is a de-duplicated list of questions
    '''
    dataset = {qt: [] for qt in QUESTION_TYPES}
    for i in tqdm(range(len(original_data))):
        item = original_data[i]
        question = item["question_entity_id"]
        answers = item["answer_ids"]
        for answer in answers:
            if question in lookup_dict and answer in lookup_dict[question]:
                relation = lookup_dict[question][answer]
                reversed = False
            elif answer in lookup_dict and question in lookup_dict[answer]:
                relation = lookup_dict[answer][question]
                reversed = True
            else:
                assert False
            
            for qt in QUESTION_TYPES:
                reduced_item = {}
                reduced_item['question'] = item['question'][qt]
                reduced_item['category'] = rel2idx[relation]
                dataset[qt] += [reduced_item]

        for qt in QUESTION_TYPES:
            json.dump(dataset[qt], open(f'step2_train_rel_model/{split}_dataset_{qt}.json', 'w'))

# if __name__ == '__main__':
#     make(train_data, 'train')
#     make(valid_data, 'valid')

def remake(original_data, split='valid', qt='KoRC-T'):
    import numpy as np
    dedup_id = np.array(json.load(open('step2_train_rel_model/valid_dedup_id.json', 'r')))
    reference = json.load(open('step2_train_rel_model/valid_dataset_KoRC-H.json', 'r'))
    dataset = []
    for i in tqdm(range(len(reference))):
        idx = np.searchsorted(dedup_id, i, side='right') - 1
        reduced_item = {}
        reduced_item['question'] = original_data[idx]['question'][qt]
        reduced_item['category'] = reference[i]['category']
        dataset += [reduced_item]
    json.dump(dataset, open(f'step2_train_rel_model/{split}_dataset_{qt}.json', 'w'))
    
remake(valid_data, 'valid')