rel2idx = {}
idx2rel = []

with open('data/wikidata5m_relation.txt', 'r') as f:
    for idx, line in enumerate(f.readlines()):
        rel = line.split('\t', 1)[0]
        rel2idx[rel] = idx
        idx2rel += [rel]

NUM_RELS = len(rel2idx)

if __name__ == '__main__':
    print(NUM_RELS)