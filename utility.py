"""
A module containing utility functions such as ...
"""

import numpy as np
from scipy.spatial.distance import squareform, pdist

def weight_to_similarity(G):
    """
    Changes the weight from a distance metric to a similarity metric
    by doing x -> 1 / (x + 1).
    """
    for v in G.node:
        for u in G[v]:
            G[v][u]['weight'] = 1 / (G[v][u]['weight'] + 1)
    return G

def euclidean_similarity_matrix(X):
    """
    Creates an `N` x `N` euclidean similarity matrix
    where `S[i, j]` is similarity between `i` and `j`.
    args:
        - `X`: the `N` x `d` feature matrix.
    returns:
        - `S`: `N` x `N` similarity matrix.
    Formula used for similarity matrix is `e^(-d/standard_deviation)`
    where `d` is euclidean distance.
    """
    D = squareform(pdist(X, 'euclidean'))

    std = np.std(D[D > 1])
    D[D > 1] = np.inf
    D = np.exp(-D / std) - np.eye(D.shape[0])

    return D
