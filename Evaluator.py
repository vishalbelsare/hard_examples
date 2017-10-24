"""
This module contains an abstract Evaluator class, which includes methods to
score a ranked set of either negative or positive hard examples.
"""


class Evaluator(object):
    """
    An abstract Evaluator class, ussed to compare the ``hardness'' of found
    examples (pairs of data points), both negative and positive by an
    algorithm.
    """

    def __init__(self, data):
        if data is None:
            raise ValueError('data argument of Evaluator object is not filled.')
        if isinstance(data, tuple) is False or len(data) != 2:
            raise ValueError('data should be a tuple of size 2.')
        self._x, self._y = data

    def evaluate(self, **kwargs):
        """
        Evaluates a list of positive or negative pairs.
        """
        raise NotImplementedError
