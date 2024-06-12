import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection

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

    # Johnson Lindenstrauss Lemma for dimensionality reduction
    dims = johnson_lindenstrauss_min_dim(n_samples=x.shape[0], eps=[0.8, 0.4, 0.18]) # Different eps values
    """
    chameleon
    orig: (2277, 3132) n_comp: [ 207  527 2169]
    crocodile
    orig: (11631, 13183) n_comp: [ 250  638 2626]
    squirrel
    orig: (5201, 3148) n_comp: [ 229  583 2400]
    """
    print(f"{dataset}\norig: {x.shape} n_comp: {dims}")
    for i, eps in enumerate([0.8, 0.4, 0.18]):
        transformer = SparseRandomProjection(dims[i], eps=eps, random_state=42)
        X_new = transformer.fit_transform(x)
        x_embedded = TSNE(n_components=2).fit_transform(X_new)
        plt.figure()
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], cmap='RdBu', c=y,  marker='o', s=10)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.colorbar()
        plt.title(f"TSNE plot of {dataset} dataset - Cont labels\neps = {eps} | dims = {dims[i]}")
        plt.savefig(f".\\img\\JL_TSNE_cont_{dataset}_eps-{eps}_dims-{dims[i]}.png")
        plt.close("all")
