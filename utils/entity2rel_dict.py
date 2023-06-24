# Function to perform the operation
def create_lookup_dict(database_file):
    lookup_dict = {}

    with open(database_file, 'r') as file:
        for line in file:
            entity, predicate, value = line.strip().split('\t')

            # Check if the entity exists in the dictionary
            if entity not in lookup_dict:
                lookup_dict[entity] = {}

            # Add the relation to the dictionary
            lookup_dict[entity][value] = predicate

    return lookup_dict

def get_lookup_dict():
    try:
        import pickle
        with open('utils/dict', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    # Create the indexed lookup dictionary
    database_file = 'data/wikidata5m_all_triplet.txt'
    lookup_dict = create_lookup_dict(database_file)

    import pickle
    with open('utils/dict', 'wb') as f:
        pickle.dump(lookup_dict, f)