import torch
import random
import numpy as np
from sklearn import preprocessing 
from matplotlib import pyplot as plt

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