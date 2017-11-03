"""
A module with functions that load the datasets.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from scipy.spatial.distance import euclidean, pdist, squareform
from utility import euclidean_similarity_matrix

def position_map(X):
    """Returns the position (pos[v]) of each vertex (v) as a dictionary"""
    pos = {}
    for i in range(X.shape[0]):
        pos[i] = (float(X[i, 0]), float(X[i, 1]))
    return pos

def knn_graph(X, k_neighbors, weighted=True, similarity=False):
    """
    Returns a directed graph where each datapoint is connected to its *k*-nearest
    other datapoints.
    """
    # A is a sparse scipy matrix.
    A = kneighbors_graph(X, k_neighbors)

    # Calculating euclidean distance between connected pairs.
    rows, cols = A.nonzero()
    for v, u in zip(rows, cols):
        A[v, u] = float(euclidean(X[v, :], X[u, :]))

    # Creating the undirected graph.
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())

    # Setting euclidean distance as edge weight.
    if weighted is True:
        if similarity is True:
            W = euclidean_similarity_matrix(X)
        else:
            W = squareform(pdist(X, 'euclidean'))
        for v in G.node:
            for u in G[v]:
                G[v][u]['weight'] = float(W[v][u])

    return G

def epsilon_ball_graph(X, epsilon, weighted=True):
    """
    Returns an undirected graph where each datapoint is connected to other
    datapoints at most **_epsilon_** distance away.
    """

    # Calculating all pair-wise distances.
    pairwise_dis = squareform(pdist(X, 'euclidean'))

    # Creating adjacency matrix based on distance matrix.
    A = np.zeros_like(pairwise_dis)
    A[pairwise_dis <= epsilon] = 1
    # Removing self-loops.
    A -= np.eye(A.shape[0])

    # Creating the graph.
    G = nx.from_numpy_matrix(A)

    # Setting euclidean distance as edge weight.
    if weighted is True:
        for v in G.node:
            for u in G[v]:
                G[v][u]['weight'] = float(G[v][u]['weight'])

    return G

def iris(remove_first_class=False, use_PCA=False):
    """Loads IRIS dataset"""
    X, y = load_iris(return_X_y=True)
    if remove_first_class:
        X = X[y != 0, :]
        y = y[y != 0]
    if use_PCA is True:
        X = PCA(n_components=2).fit_transform(X)
    return X, y
