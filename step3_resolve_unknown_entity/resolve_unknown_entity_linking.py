import spacy
import json
from tqdm import tqdm
from step3_resolve_unknown_entity.resolve_unknown_entity_dp import run
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

entity_linker = spacy.load("en_core_web_lg")
entity_linker.add_pipe("entityLinker", last=True)

def naive_string_similarity(string1, string2):
    # Tokenize strings and create vocabulary
    vectorizer = CountVectorizer()
    strings = [string1, string2]
    X = vectorizer.fit_transform(strings)

    # Compute cosine similarity
    similarity = cosine_similarity(X)[0, 1]

    return similarity

def resolve_by_sent_link(x, y):
    doc_x = entity_linker(x)
    doc_y = entity_linker(y)

    for sent_x, sent_y in zip(doc_x.sents, doc_y.sents):
        if '[' in sent_y.text and ']' in sent_y.text:
            item_x = sent_x._.linkedEntities
            item_y = sent_y._.linkedEntities
            masked_description = sent_y.text[sent_y.text.index('[') + 1: sent_y.text.index(']')]
            id_y = set([item.identifier for item in item_y])
            item_x_minus_y = [item for item in item_x if item.identifier not in id_y]
            max_sim = -1
            if not item_x_minus_y:
                return '', None
            for item in item_x_minus_y:
                sim = naive_string_similarity(item.description, masked_description) if item.description else 0
                if sim > max_sim:
                    max_sim = sim
                    resolved_item = item
            return resolved_item.label, resolved_item.identifier
    return '', None

if __name__ == '__main__':
    for data_split in ['valid', 'train']:
        with open(f'step3_resolve_unknown_entity/{data_split}_prompt.txt', 'r') as f:
            all_questions = f.read().split('\n\n')
        data = json.load(open(f"data/{data_split}.json", 'r'))
        assert(len(data) == len(all_questions))
        entity_acc = 0
        all_answers = []

        pbar = tqdm(range(len(all_questions)))
        for idx in pbar:
            try:
                x, y, _ = all_questions[idx].split('\n')
            except ValueError:
                all_answers += ['']
                continue
            entity, entity_id = resolve_by_sent_link(x, y)
            all_answers += [entity]
            oracle = data[idx]['question_entity']
            oracle_id = int(data[idx]['question_entity_id'][1:])
            if entity_id == oracle_id:
                entity_acc += 1
            else:
                _ = 'stop'
            pbar.set_postfix_str('acc: ' + str(round(entity_acc / (idx + 1), 2)))
        print('acc: ', entity_acc / len(all_questions))
        with open(f'step3_resolve_unknown_entity/{data_split}_answers_linking_new.txt', 'w') as f:
            f.write('\n'.join(all_answers))