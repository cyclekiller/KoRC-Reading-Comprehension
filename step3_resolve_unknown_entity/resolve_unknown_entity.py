from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

model_name = 'google/flan-t5-base' # 'google/flan-t5-xl'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 2048
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
print('Language model loaded.')

for data_split in ['train', 'valid']:
    with open(f'step3_resolve_unknown_entity/{data_split}_prompt.txt', 'r') as f:
        all_questions = f.read().split('\n\n')
    all_answers = []
    batch_size = 4
    num_batches = (len(all_questions) - 1) // batch_size + 1
    for i in tqdm(range(num_batches)):
        batch_questions = []
        errors = []
        for idx in range(i * batch_size, min(len(all_questions), (i + 1) * batch_size)):
            my_question = all_questions[idx]
            if my_question.startswith('Error parsing the text'):
                errors += [idx]
                continue
            batch_questions += [my_question]
        tokens = tokenizer(batch_questions, padding=True, truncation=False, return_tensors='pt').to('cuda')
        outputs = model.generate(**tokens).cpu()
        answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for idx in errors:
            answers.insert(idx - i * batch_size, "")
        all_answers += answers
        for idx in errors:
            assert all_answers[idx] == ""
    with open(f'step3_resolve_unknown_entity/{data_split}_answers_prompt.txt', 'w') as f:
        f.write('\n'.join(all_answers))