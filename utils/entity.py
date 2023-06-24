import pickle
from spacy_entity_linker.DatabaseConnection import get_wikidata_instance

def create_alias_dict():
    entity_alias_dict ={}
    with open('data/wikidata5m_entity.txt') as f:
        for line in f.readlines():
            if line.find('Q719659\t') >= 0:
                _ = 'stop'
            line = line.strip()
            tmp = line.split('\t')
            entity = tmp[0]
            aliases = tmp[1:]
            for alias in aliases:
                assert alias not in entity_alias_dict, (alias, entity_alias_dict[alias], entity)
                entity_alias_dict[alias] = entity

    pickle.dump(entity_alias_dict, open('utils/alias_dict', 'wb'))

def entity2label(entity_id):
    label = wikidata.get_entity_name(entity_id)
    if label == '<none>':
        # TODO: get an alias for this entity
        _ = 'stop'
    return label

if __name__ == '__main__':
    create_alias_dict()
else:
    alias2entity_dict = pickle.load(open('utils/alias_dict', 'rb'))
    wikidata = get_wikidata_instance()
