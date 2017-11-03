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
        self.graph = knn_graph(self._data, self.k_neighbors, similarity=True)
        self.ego_radius = ego_radius

    def find_positive(self, k, *args):
        pass

    def find_negative(self, k, *args):
        pass

    def find_examples(self, k):
        """
        Function that finds hard examples, both positive *and* negative.
        """
        hard_edges = []
        scores = []
        for center in self.graph.nodes:
            members, center_examples = self._find_cut(center)
            hard_edges.extend(center_examples)
            scores.append(self._conductance(members))

        return list(set(hard_edges)), scores


    def _find_cut(self, center):
        """
        Finds a spectral cut in the ego-network of `node`.
        args:
            - `center`: center of ego-network.
        returns:
            - `members`: two sets of nodes, for the two parts of the cut.
            - `inner_edges`: edges between two clusters connected to the center node.
        """
        ego = nx.ego_graph(self.graph, center, radius=self.ego_radius)
        spectral = SpectralClustering(n_clusters=2,
                                      affinity='precomputed',
                                      eigen_solver='arpack')

        spectral.fit(nx.adjacency_matrix(ego))

        # Members of each cluster are saved in this list.
        members = [np.array(ego.nodes) \
            [np.where(spectral.labels_ == cls)[0]] for cls in [0, 1]]

        # Inter-cluster links are returned also.
        place = dict()
        for index, node in enumerate(ego.nodes):
            place[node] = index
        inter_edges = [(v, u) for (v, u) in ego.edges() \
            if spectral.labels_[place[v]] != spectral.labels_[place[u]] and \
            (v == center or u == center)]

        return members, inter_edges

    def _conductance(self, members):
        """Returns conductance of a graph cut"""
        return nx.conductance(self.graph, members[0], members[1])
