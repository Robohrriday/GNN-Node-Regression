import pandas as pd
import torch
from torch_geometric.data import Data
import os
import json

"""
Extracts data from the wikipedia datasets and saves it in the processed folder.
The data is saved in the processed folder in the form of a .pt file.

Data object description:
    x: Node features (torch.Tensor): Node feature matrix with shape [num_nodes, max(feature value among all nodes)+1]. 
    It is binary array where each row represents a node and each column represents a feature. If a node has a feature, the corresponding column value is 1, else 0.

    edge_index: Edge indices (torch.Tensor): Undirected graph edge indices with shape [2, 2*num_edges].

    y: Target (torch.Tensor): Target node labels with shape [num_nodes].
"""


datasets = ["chameleon", "crocodile", "squirrel"]
for dataset in datasets:
    path = str(os.path.curdir) + "\\data\\wikipedia\\" + dataset + "\\"
    edges = pd.read_csv(path + "musae_" + dataset + "_edges.csv")
    file = open(path + "musae_" + dataset + "_features.json")
    features = json.load(file)
    target = pd.read_csv(path + "musae_" + dataset + "_target.csv")
    edge_index = torch.tensor(edges.to_numpy(), dtype=torch.long).t().contiguous()
    edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
    max_val = 0
    for node in target["id"]:
        max_val = max(max_val, max(features[str(node)]))
    x = torch.zeros(len(target["id"]), max_val+1, dtype=torch.long)
    for node in target["id"]:
        x[node, features[str(node)]] = 1
    y = torch.tensor(target["target"].values, dtype = torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    target_path = str(os.path.curdir) + f"\\processed\\{dataset}\\"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    torch.save(data, target_path + "processed.pt")