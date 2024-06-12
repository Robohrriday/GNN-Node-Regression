import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

datasets = ["chameleon", "crocodile", "squirrel"]
for dataset in datasets:
    path = str(os.path.curdir) + "\\processed\\" + dataset + "\\"
    data = torch.load(path + "processed.pt")
    x = data.x.numpy()
    y = data.y.numpy()
    # Convert continuous labels to discrete labels (10 bins)
    x_embedded = TSNE(n_components=2).fit_transform(x)
    plt.figure()
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], cmap='RdBu', c=y,  marker='o', s=10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar()
    plt.title(f"TSNE plot of {dataset} dataset - Continuous labels")
    plt.savefig(f".\\img\\TSNE_cont_{dataset}.png")
    y = np.digitize(y, bins=np.linspace(y.min(), y.max(), 10))
    plt.figure()
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], cmap='RdBu', c=y,  marker='o', s=10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar()
    plt.title(f"TSNE plot of {dataset} dataset - Discrete labels")
    plt.savefig(f".\\img\\TSNE_disc_{dataset}.png")

    ## Perform PCA on the data and then plot the data
    for n_comp in [2, 5, 10, 20, 50, 100]:
        pca = PCA(n_components=n_comp)
        x_embedded = pca.fit_transform(x)
        plt.figure()
        x_embedded = TSNE(n_components=2).fit_transform(x_embedded)
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], cmap='RdBu', c=y,  marker='o', s=10)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.colorbar()
        plt.title(f"TSNE plot of {dataset} dataset - Discrete labels - {n_comp} components")
        plt.savefig(f".\\img\\TSNE_disc_{dataset}_{n_comp}_components.png")

    


