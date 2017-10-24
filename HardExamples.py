"""
This module contains an abstract HardExamples class, which given a
dataset (X,y) finds ``hard'' positive and negative examples.
"""

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

    def find_positive(self, k):
        """
        Function that finds the top k hardest positive examples.
        args:
            k (int): Number of examples returned.
        Returns:
            a list of k hardest positive examples.
        """
        raise NotImplementedError

    def find_negative(self, k):
        """
        Function that finds the top k hardest negative examples.
        args:
            k (int): Number of examples returned.
        Returns:
            a list of k hardest negative examples.
        """
        raise NotImplementedError
