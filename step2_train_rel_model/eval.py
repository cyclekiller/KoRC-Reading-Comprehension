import json
import torch
from train import *

def calculate_accuracy(data_loader):
    correct = 0
    total = 0
    all_predicted = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            outputs = my_model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            all_predicted += predicted.cpu().numpy().tolist()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return np.array(all_predicted), accuracy

if __name__ == '__main__':
    for train_qt in QUESTION_TYPES:
        checkpoint = torch.load(f'step2_train_rel_model/epoch=29-step=31950_{train_qt}.ckpt')
        my_model.load_state_dict(checkpoint['state_dict'])
        my_model.cuda()
        my_model.eval()

        for eval_qt in QUESTION_TYPES:
            data_split = 'valid'
            dataset = get_datasets(data_split, eval_qt, eval=True)
            preds, accuracy = calculate_accuracy(DataLoader(dataset, batch_size=64, shuffle=False))
            print(f"Evaluate {train_qt} model on {eval_qt}. Accuracy: {accuracy}")
            '''
            Evaluate KoRC-H model on KoRC-H. Accuracy: 0.8476280696640677
            Evaluate KoRC-H model on KoRC-T. Accuracy: 0.7378853020476572
            Evaluate KoRC-H model on KoRC-L. Accuracy: 0.6891714639656051

            Evaluate KoRC-T model on KoRC-H. Accuracy: 0.6627195219704146
            Evaluate KoRC-T model on KoRC-T. Accuracy: 0.9474240326459229
            Evaluate KoRC-T model on KoRC-L. Accuracy: 0.5517015229905997

            Evaluate KoRC-L model on KoRC-H. Accuracy: 0.7761058077679808
            Evaluate KoRC-L model on KoRC-T. Accuracy: 0.7938861764920206
            Evaluate KoRC-L model on KoRC-L. Accuracy: 0.8277344603949573
            '''
            if train_qt == DEFAULT_QUESTION_TYPE and eval_qt == DEFAULT_QUESTION_TYPE:
                dedup_id = np.array(json.load(open(f'step2_train_rel_model/{data_split}_dedup_id.json', 'r')), dtype=np.intp)
                with open(f'step2_train_rel_model/valid_predicted.txt', 'w') as f:
                    for pred in preds[dedup_id]:
                        print(idx2rel[pred], file=f)
else:
    checkpoint = torch.load(f'step2_train_rel_model/epoch=29-step=31950_submit.ckpt')
    my_model.load_state_dict(checkpoint['state_dict'])
    my_model.cuda()
    my_model.eval()
    print(f'Relation classification model loaded.')
