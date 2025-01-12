# Graph Attention Network
PyTorch implementation of the paper Graph Attention Networks

### Overview
This repository contains a custom implementation of the Graph Attention Network (GAT) as described in the original research [paper](https://arxiv.org/abs/1710.10903), applied to the Cora citation network dataset. The project demonstrates node classification using attention mechanisms in graph neural networks.

### Project Structure
* Cora EDA.ipynb: Exploratory Data Analysis notebook
* layer.py: Custom GAT layer implementation
* train.py: Training script for the GAT model

## Dataset
The Cora dataset is a citation network where nodes represent scientific papers, and edges represent citations between papers.

## Model Architecture
The Graph Attention Network uses a self-attention mechanism to aggregate node features, allowing the model to learn adaptive importance of neighboring nodes.

## Training the Model
```python train.py```

Licence MIT
