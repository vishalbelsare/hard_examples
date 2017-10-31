"""
This module contains the CycleExamples class, which finds hard examples
according to the paper at:
http://faculty.ucmerced.edu/mhyang/papers/eccv16_feature_learning_supp.pdf
"""

from HardExamples import HardExamples
from datasets import knn_graph
import numpy as np
import networkx as nx

class CycleExamples(HardExamples):
    """
    This class finds hard positive and negatives based on the method described
    in:
    http://faculty.ucmerced.edu/mhyang/papers/eccv16_feature_learning_supp.pdf
    """
    def __init__(self, data, k_neighbors):
        super(CycleExamples, self).__init__(self, data)
        self.k_neighbors = k_neighbors
        self.graph = knn_graph(self._data, self.k_neighbors)

    def _edge_cycle(self, edge, ell):
        """
        Finds whether or not a link is in a directed cycle of length ell

        **params**:
            - edge: tuple of (source, target)
            - ell: length of the cycle

        **return**
            - `True`/`False`: whether there is a cycle with this link shorter than `ell`.
            - length of the cycle.
        """
        v, u = edge
        # 1. We remove the v --> u link.
        self.graph.remove_edge(v, u)

        # 2. We find a directed path of length at most ell-1 from u to v.
        try:
            u_v_dis = nx.shortest_path_length(self.graph, source=u, target=v)
        except nx.NetworkXNoPath:
            # There is no path between u and v.
            return False, np.inf
        return (u_v_dis + 1 <= ell), u_v_dis + 1

    def find_positive(self, k):
        pass

    def find_negative(self, k):
        pass
