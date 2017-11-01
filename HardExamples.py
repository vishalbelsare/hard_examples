"""
This module contains an abstract HardExamples class, which given a
dataset (X,y) finds ``hard'' positive and negative examples.
"""

from scipy.spatial.distance import euclidean

class HardExamples(object):
    """
    An abstract class for hard example finding methods.
    """

    def __init__(self, data):
        if data is None:
            raise ValueError('data argument of Evaluator object is not filled.')
        if isinstance(data, tuple) is False or len(data) != 2:
            raise ValueError('data should be a tuple of size 2.')
        self._data, self._label = data

    def find_positive(self, k=-1, *args):
        """
        Function that finds the top k hardest positive examples.
        args:
            k (int): Number of examples returned. If -1 given, all examples are returned.
        Returns:
            a list of k hardest positive examples.
        """
        raise NotImplementedError

    def find_negative(self, k=-1, *args):
        """
        Function that finds the top k hardest negative examples.
        args:
            k (int): Number of examples returned. If -1 given, all examples are returned.
        Returns:
            a list of k hardest negative examples.
        """
        raise NotImplementedError
    
    def _distance(self, pair, metric='euclidean'):
        v, u = pair
        if metric == 'euclidean':
            return euclidean(self._data[v, :], self._data[u, :])
        else:
            raise NotImplementedError

    def _rank_negative(self, pairs):
        """
        Ranks negative examples based on their *hardness*.
        Pairs are sorted based on their distance (euclidean by default), closer
        pairs are *harder* negative examples.
        """
        return sorted(pairs, key=self._distance)

    def _rank_positive(self, pairs):
        """
        Ranks positive examples based on their *hardness*.
        Pairs are sorted based on their distance (euclidean by default), farther
        pairs are *harder* positive examples.
        """
        return sorted(pairs, key=self._distance, reverse=True)
