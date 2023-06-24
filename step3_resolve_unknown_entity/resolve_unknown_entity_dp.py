import json
from tqdm import tqdm
from lcs_cython import lcs
from spacy.lang.en import English
tokenizer = English().tokenizer

def rev_lcs(x, y):
    return [item[::-1] for item in reversed(lcs(x[::-1], y[::-1]))]

def run(x, y):
    y = y.replace(' .', '. ')
    try:
        left_y = y.index('[')
        right_y = y.index(']')
    except ValueError:
        return ''

    x = [token.text for token in tokenizer(x)]
    pre_span = [token.text for token in tokenizer(y[:left_y])]
    post_span = [token.text for token in tokenizer(y[right_y + 1:])]

    f = lcs(x, pre_span)
    rev_f = rev_lcs(x, post_span)

    max_score = 0
    ki = kj = None
    for i in range(len(x) + 1):
        for j in range(i, len(x) + 1):
            if f[i][-1] + rev_f[j][0] > max_score:
                ki, kj = i, j
                max_score = f[i][-1] + rev_f[j][0]
            elif ki == i and f[i][-1] + rev_f[j][0] == max_score:
                kj = j
    return ' '.join(x[ki: kj]).replace(' - ', '-').replace(' , ', ', ')

if __name__ == '__main__':
    for data_split in ['valid', 'train']:
        with open(f'step3_resolve_unknown_entity/{data_split}_prompt.txt', 'r') as f:
            all_questions = f.read().split('\n\n')
        data = json.load(open(f"data/{data_split}.json", 'r'))
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
        with open(f'step3_resolve_unknown_entity/{data_split}_answers_dp.txt', 'w') as f:
            f.write('\n'.join(all_answers))