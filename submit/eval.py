import json
from step3_resolve_unknown_entity.resolve_unknown_entity_linking import *
from utils.entity import *
import pickle
from tqdm import tqdm

def create_one_hop_dict(database_file):
    one_hop_dict, one_hop_reverse_dict = {}, {}
    with open(database_file, 'r') as file:
        for line in file:
            entity, predicate, value = line.strip().split('\t')

            # Check if the entity exists in the dictionary
            if entity not in one_hop_dict:
                one_hop_dict[entity] = {}
            
            if predicate not in one_hop_dict[entity]:
                one_hop_dict[entity][predicate] = set()

            # Add the relation to the dictionary
            one_hop_dict[entity][predicate].add(value)

            # Add the reverse relation
            if value not in one_hop_reverse_dict:
                one_hop_reverse_dict[value] = {}
            if predicate not in one_hop_reverse_dict[value]:
                one_hop_reverse_dict[value][predicate] = set()
            one_hop_reverse_dict[value][predicate].add(entity)

    pickle.dump(one_hop_dict, open('submit/one_hop_dict', 'wb'))
    pickle.dump(one_hop_reverse_dict, open('submit/one_hop_reverse_dict', 'wb'))
# create_one_hop_dict('data/wikidata5m_all_triplet.txt')

final_eval = True # False
if final_eval:
    one_hop_dict = pickle.load(open('submit/one_hop_dict', 'rb'))
    one_hop_reverse_dict = pickle.load(open('submit/one_hop_reverse_dict', 'rb'))
    print('one-hop dict loaded.')
    
def compare(my_answer, answer):
    if len(my_answer) == 0:
        return 0, 0
    correct = 0
    for item in answer:
        if item in my_answer:
            correct += 1
    return correct / len(answer), correct / len(my_answer)

def get_entity_id(my_answer):
    answer = my_answer.strip()
    if answer in alias2entity_dict:
        entity_id = alias2entity_dict[answer]
    else:
        linking_result = entity_linker(answer)._.linkedEntities
        if linking_result:
            entity_id = f'Q{entity_linker(answer)._.linkedEntities[0].identifier}'
        else:
            return None
    return entity_id

def get_final_answer(entity_id, my_rel) -> set:
    my_rel = my_rel.strip()
    try:
        final_answer = one_hop_dict[entity_id][my_rel]
    except KeyError:
        try:
            final_answer = one_hop_reverse_dict[entity_id][my_rel]
        except KeyError:
            final_answer = set()
    return final_answer

if __name__ == '__main__':
    for data_split in ['valid', 'train']:
        with open(f'step3_resolve_unknown_entity/{data_split}_answers_dp.txt', 'r') as f:
            my_answers = f.readlines()
        with open(f'step3_resolve_unknown_entity/{data_split}_prompt.txt', 'r') as f:
            all_questions = f.read().split('\n\n')
        data = json.load(open(f"data/{data_split}.json", 'r'))
        my_rels = open(f'step2_train_rel_model/{data_split}_predicted.txt', 'r').readlines()
        assert len(my_answers) == len(data) == len(my_rels)
        entity_acc = final_precision = final_recall = 0
        for i in tqdm(range(len(my_answers))):
            entity_id = get_entity_id(my_answers[i])
            if entity_id == data[i]['question_entity_id']:
                entity_acc += 1
                if final_eval:
                    final_answer = get_final_answer(entity_id, my_rels[i])
                    precision, recall = compare(final_answer, data[i]['answer_ids'])
                    final_precision += precision
                    final_recall += recall

        print(f'{data_split} question_entity accuracy: {entity_acc / len(data)}') # prompt: ~50%, dp: ~60%, dp + linking: 68%
        if final_eval:
            print(f'{data_split} final precision/recall: {final_precision / len(data)}, {final_recall / len(data)}')
# valid question_entity accuracy: 0.6819382096646422
# valid final precision/recall: 0.5805606111938123, 0.5849275101160569
# train question_entity accuracy: 0.6758511480601742
# train final precision/recall: 0.6338530973063689, 0.6363645442794482