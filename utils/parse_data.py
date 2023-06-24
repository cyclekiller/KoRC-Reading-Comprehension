import json

def parse_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Parse train.json
train_data = parse_json_file('data/train.json')

# Parse valid.json
valid_data = parse_json_file('data/valid.json')
