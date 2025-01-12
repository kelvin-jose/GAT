import torch
import random
import numpy as np
from sklearn import preprocessing 
from matplotlib import pyplot as plt

CORA_TARGET_LOCATION = '../datasets/cora_preprocessed'
EDGE_INFO = 'cora_edge_index.npy'
FEATURES = 'cora_features.npy'
LABELS = 'cora_labels.npy'