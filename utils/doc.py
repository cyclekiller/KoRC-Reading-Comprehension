doc_to_entity = []
docs = []

with open('data/wikidata5m_text.txt', 'r') as f:
    for line in f:
        tmp = line.strip().split('\t')
        entity = tmp[0]
        texts = tmp[1:]
        docs += texts
        doc_to_entity += [entity] * len(texts)

print(f"num of docs: {len(docs)}")