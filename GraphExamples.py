"""
This module contains our Graph-based methods for getting hard examples.
"""

from HardExamples import HardExamples

class GraphExamples(HardExamples):
    """
    A graph-cut based method to retrieve hard examples.
    """

    def __init__(self, data):
        super(GraphExamples, self).__init__(self, data)

    def find_positive(self, k):
        pass

    def find_negative(self, k):
        pass
