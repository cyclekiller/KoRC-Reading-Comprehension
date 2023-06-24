import numpy as np
import pandas as pd
from collections import defaultdict
from utils.constants import *
from tqdm import tqdm

def calc_bert_features(data_split, question_type=DEFAULT_QUESTION_TYPE):
    df = pd.read_json(f"step2_train_rel_model/{data_split}_dataset_{question_type}.json", orient='records')
    unique_questions = df["question"].unique()
    unique_questions_to_id = {question: idx for idx, question in enumerate(unique_questions)}
    repeated_indices = defaultdict(list)
    for i in range(len(df)):
        repeated_indices[unique_questions_to_id[df.loc[i, "question"]]] += [i]
    repeated_indices = {idx: np.array(item) for idx, item in repeated_indices.items()}        
    tokenized = tokenizer(unique_questions.tolist() , padding = True, truncation = True,  return_tensors="pt")
    batch_size = 128
    num_batches = (tokenized['input_ids'].size(0) - 1) // batch_size + 1
    with torch.no_grad():
        x = torch.empty([len(df), 768]).to('cuda')
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, tokenized['input_ids'].size(0))
            batch_tokenized = {k: v[start_idx: end_idx].to('cuda') for k, v in tokenized.items()}
            batch_hidden = model(**batch_tokenized).last_hidden_state[:, 0, :]
            for idx in range(start_idx, end_idx):
                x[repeated_indices[idx]] = batch_hidden[idx - start_idx]
    x = x.to("cpu").numpy()
    np.save(open(f'step2_train_rel_model/{data_split}_data_bert_feature_{question_type}.npy', 'wb'), x, allow_pickle=False)

if __name__ == '__main__':
    import torch
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to('cuda')
    for data_split in ['valid', 'train']:
        for question_type in QUESTION_TYPES:
            calc_bert_features(data_split, question_type)

