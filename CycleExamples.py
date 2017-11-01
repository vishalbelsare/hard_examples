"""
This module contains the CycleExamples class, which finds hard examples
according to the paper at:
http://faculty.ucmerced.edu/mhyang/papers/eccv16_feature_learning_supp.pdf
"""

import random
import numpy as np
import networkx as nx
from HardExamples import HardExamples
from datasets import knn_graph
from scipy.spatial.distance import pdist, squareform

class CycleExamples(HardExamples):
    """
    This class finds hard positive and negatives based on the method described
    in:
    http://faculty.ucmerced.edu/mhyang/papers/eccv16_feature_learning_supp.pdf
    """
    def __init__(self, data, k_neighbors):
        super(CycleExamples, self).__init__(data)
        self.k_neighbors = k_neighbors
        self.graph = knn_graph(self._data, self.k_neighbors)

    def _edge_cycle(self, edge):
        """
        Returns the length of the smallest directed cycle `edge` is a part of.

        :args:
            - `edge`: tuple of (source, target)

        :return: length of the cycle.
        """
        v, u = edge
        # 1. We remove the v --> u link.
        try:
            edge_data = self.graph.get_edge_data(v, u)
            self.graph.remove_edge(v, u)
        except nx.NetworkXError:
            return np.inf

        # 2. We find a directed path of length at most ell-1 from u to v.
        try:
            # cycle_length = reverse_path_length + 1
            u_v_dis = nx.shortest_path_length(self.graph, source=u, target=v) + 1
        except nx.NetworkXNoPath:
            # No path from u to v.
            u_v_dis = np.inf
        # Put the edge back in place and return results.
        self.graph.add_edge(v, u)
        for attr, val in edge_data.items():
            self.graph[v][u][attr] = val
        return u_v_dis

    # Running time of this function is O(E^2), if E=O(N), it's O(N^2).
    def find_positive(self, k=-1, ell=-1, *args):
        """
        :args:
            - `k`: the number of examples returned. `-1` returns all.
            - `ell`: Maximum length of the cycle a positive edge is a member of.
            `-1` means all cycles are considered.

        :return: list of negative pairs.
        """
        res = []
        for edge in self.graph.edges():
            cycle_length = self._edge_cycle(edge)
            if cycle_length <= ell or (cycle_length < np.inf and ell == -1):
                res.append(edge)

        if k == -1:
            # Returning all instances.
            return res
        return self._rank_positive(res)[:k]

    def find_negative(self, k=-1, threshold=1.0, *args):
        """
        :args:
            - `k`: number of examples returned. `-1` returns all.
            - `threshold`: pairs longer than this threshold are returned.

        :return: list of negative pairs.
        """

        # Calculating pairwise geodesic distance.
        geodesic_dis = {source:targets for source, targets \
            in nx.all_pairs_dijkstra_path_length(self.graph)}

        res = []
        for source, reachable_targets in geodesic_dis.items():
            for target, distance in reachable_targets.items():
                if distance > threshold:
                    res.append((source, target))

        if k == -1:
            return res
        return random.sample(res, k)
