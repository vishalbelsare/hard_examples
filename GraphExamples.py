"""
This module contains our Graph-based methods for getting hard examples.
"""

import numpy as np
import networkx as nx
from HardExamples import HardExamples
from datasets import knn_graph
from utility import weight_to_similarity
from sklearn.cluster import SpectralClustering

class GraphExamples(HardExamples):
    """
    A graph-cut based method to retrieve hard examples.
    """

    def __init__(self, data, k_neighbors, ego_radius=2):
        super(GraphExamples, self).__init__(data)
        self.k_neighbors = k_neighbors
        self.graph = knn_graph(self._data, self.k_neighbors)
        self.graph = weight_to_similarity(self.graph)
        self.ego_radius = ego_radius

    def find_positive(self, k, *args):
        pass

    def find_negative(self, k, *args):
        pass
    
    def find_examples(self, k):
        """
        Function that finds hard examples, both positive *and* negative.
        """
        pass
    
    def _find_cut(self, center):
        """
        Finds a spectral cut in the ego-network of `node`.
        args:
            - `center`: center of ego-network.
        returns: two sets of nodes, for the two parts of the cut.
        """
        ego = nx.ego_graph(self.graph, center, radius=self.ego_radius)
        spectral = SpectralClustering(n_clusters=2,
                                      affinity='precomputed',
                                      eigen_solver='arpack')

        spectral.fit(nx.adjacency_matrix(ego))
        members = [np.where(spectral.labels_ == cls) for cls in [0, 1]]

        return members
