"""
This module contains an Evaluator class, which includes methods to score
a ranked set of either negative or positive hard examples.
"""


class Evaluator(object):
    """
    An Evaluator class, ussed to compare the ``hardness'' of found examples
    (pairs of data points), both negative and positive by an algorithm.
    """

    def __init__(self):
        self._x = None
        self._y = None

    def set_dataset(self, data):
        """
        data can be either a string, showing the path for the data, or a tuple
        of (X,y)
        """

        if isinstance(data, str):
            print 'data argument was a string.'
            raise NotImplementedError('Reading dataset from string is not' + \
                                      'yet implemented.')
        elif isinstance(data, tuple) and len(data) == 2:
            print 'data is a (X,y) tuple.'
            self._x, self._y = data
            #TODO: Check if the data is in float numpy arrays.
        else:
            raise  ValueError('dataset is neither a string or' + \
                              'a tuple of size 2.')

    def find_distributions(self, dataset=None):
        """
        Given a dataset, this method finds two ``normal'' distributions. One
        for positive pairs and one for negative pairs. These are used for
        later scoring of examples.
        """
        if dataset is None and (self._x is None or self._y is None):
            # We don't have a working dataset.
            raise ValueError('Dataset cannot be None.')

    @staticmethod
    def evaluate():
        """
        args:
            data: a tuple of (X,y) for the dataset used.
            neg: a list of top-k negative hard examples.
            pos: a list of top-k positive hard examples.

        Returns:
            a score from 0.0 to 1.0 showing how hard found examples are.
            Here, a higher score means better pairs.
        """
        pass
