import torch
import random
import numpy as np
from sklearn import preprocessing 
from matplotlib import pyplot as plt

from layer import GATLayer

CORA_TARGET_LOCATION = '../datasets/cora_preprocessed'
EDGE_INFO = 'cora_edge_index.npy'
FEATURES = 'cora_features.npy'
LABELS = 'cora_labels.npy'

edge_index = np.load(f'{CORA_TARGET_LOCATION}/{EDGE_INFO}')
node_features = np.load(f'{CORA_TARGET_LOCATION}/{FEATURES}')
node_labels = np.load(f'{CORA_TARGET_LOCATION}/{LABELS}', allow_pickle=True)

label_encoder = preprocessing.LabelEncoder()
labels2idx = torch.tensor(label_encoder.fit_transform(node_labels), dtype = torch.long)

seed = 52
np.random.seed(seed)

def random_stratified_train_indices(items, sample_size=20):
    unique_items = np.unique(items)
    indices = []
    for item in unique_items:
        item_indices = np.where(items == item)[0]
        if len(item_indices) > sample_size:
            sample_indices = np.random.choice(item_indices, sample_size, replace=False)
        else:
            sample_indices = item_indices
        indices.extend(sample_indices)
    return indices

train_indices = random_stratified_train_indices(node_labels)
other_indices = np.setdiff1d(range(len(node_labels)), train_indices)
val_indices = np.random.choice(other_indices, 500, replace=False)
other_indices = np.setdiff1d(range(len(other_indices)), val_indices)
test_indices = np.random.choice(other_indices, 1000, replace=False)

print('train data size: ', len(train_indices))
print('valid data size: ', len(val_indices))
print('test data size: ', len(test_indices))

train_config = {
    'num_layers': 2,
    'fin': [1433, 64],
    'num_heads': [8, 1],
    'fout': [8, 7],
    'concat': [True, False],
    'num_epochs': 1000,
    'lr': 0.005,
    'weight_decay': 0.0005
}

class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()        
        self.gat_model = torch.nn.Sequential(
              *[GATLayer(fin = train_config['fin'][layer_idx],
                        fout = train_config['fout'][layer_idx],
                        num_heads = train_config['num_heads'][layer_idx],
                        concat = train_config['concat'][layer_idx])  
               for layer_idx in range(train_config['num_layers'])])
        
model = GAT()
optimizer = torch.optim.Adam(params = model.parameters(), lr = train_config['lr'], weight_decay = train_config['weight_decay'])
loss_fn = torch.nn.CrossEntropyLoss()
X = torch.Tensor(node_features)
train_losses = []
valid_losses = []

for epoch in range(train_config['num_epochs']):
    optimizer.zero_grad()
    scores = model.gat_model(X)
    train_loss = loss_fn(scores[train_indices], labels2idx[train_indices])
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.detach().numpy())
    
    if epoch % 100 == 0:
        valid_loss = loss_fn(scores[val_indices], labels2idx[val_indices])
        pred = torch.argmax(scores[val_indices], -1) 
        accuracy = torch.sum(pred == labels2idx[val_indices]) / len(pred)
        print(f'train loss: {train_loss:.3f} | valid loss: {valid_loss:.3f} | valid accuracy: {accuracy:.3f}')
        valid_losses.append(valid_loss.detach().numpy())
