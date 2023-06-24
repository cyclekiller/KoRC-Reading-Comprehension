import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.relation import *
from step2_train_rel_model.get_bert_feature import *

NUM_LABELS = NUM_RELS  # Number of output labels

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].cuda(), self.y[idx].cuda()

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Classifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_labels, learning_rate=1e-3):
        super(Classifier, self).__init__()
        self.model = FFN(input_dim, hidden_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

my_model = Classifier(input_dim=768, hidden_dim=256, num_labels=NUM_LABELS)

def get_datasets(data_split, question_type=DEFAULT_QUESTION_TYPE, eval=False):
    x = np.load(open(f'step2_train_rel_model/{data_split}_data_bert_feature_{question_type}.npy', 'rb'))
    y = pd.read_json(f"step2_train_rel_model/{data_split}_dataset_{question_type}.json", orient='records')["category"]
    x = torch.Tensor(x).float()
    y = torch.Tensor(y).long()
    if eval:
        return MyDataset(x, y)
    else:
        y_one_hot = torch.zeros((len(y), NUM_LABELS))
        y_one_hot.scatter_(1, y.unsqueeze(1), 1)
        return MyDataset(x, y_one_hot)

if __name__ == '__main__':
    for train_qt in QUESTION_TYPES:
        train_dataset = get_datasets('train', train_qt)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        trainer = pl.Trainer(max_epochs=30)
        trainer.fit(my_model, train_loader)
        print(f'{train_qt} model trained.')
