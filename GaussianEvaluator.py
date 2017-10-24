"""
This module contains a Gaussian Evaluator class.
"""

from itertools import combinations
from scipy.stats import multivariate_normal
import numpy as np
from Evaluator import Evaluator

class GaussianEvaluator(Evaluator):
    """
    This class assumes two Gaussian distributions for both ``positive'' and
    ``negative'' pairs. Then, by comparing the likelihood of a pair of
    datapoints belonging to either group, we score the hardness of that pair.
    """

    def __init__(self, data):
        # Superclass constructor sets the datasets, if given.
        super(GaussianEvaluator, self).__init__(self, data)

        # Fitting Gaussian distributions

        positive_diffs = []
        negative_diffs = []
        # Going through every pair of data points.
        for first, second in combinations(range(self._data.shape[0])):
            diff = np.abs(self._data[first, :] - self._data[second, :])
            if self._label[first] == self._label[second]: # A positive (same-class) pair.
                positive_diffs.append(diff)
            else: # A negative (different class) pair.
                negative_diffs.append(diff)

        # Sample usage: self.prob['pos']['gaussian'] = Gaussian for positives
        self.prob = {
            'pos': {'gaussian': None, 'prior': None},
            'neg': {'gaussian': None, 'prior': None}
        }

        for pair_type, diffs in zip(['pos', 'neg'],
                                    [positive_diffs, negative_diffs]):
            self.prob[pair_type]['gaussian'] = multivariate_normal(
                np.mean(diffs, axis=0),
                np.cov(diffs, rowvar=0) # Covariance between dimensions.
            )
            # Prior = N(self)/N(total)
            self.prob[pair_type]['prior'] = float(len(diffs)) / \
                (len(positive_diffs) + len(negative_diffs))

    def belong_prob(self, diff_value, pair_type):
        """
        Calculates probability of an absolute difference value belonging to a
        pair type, positive or negative. The formula used is:
        Prob{diff_value in pair_type} = Gaussian(diff_value) * prior(pair_type)
        args:
            - diff_value: (ndarray) the absolute difference between two data points.
            - pair_type: ('pos'/'neg') prob of belonging to corresponding Gaussians.
        Returns:
            - (float in [0.0, 1.0]) The probability of the difference value
            belonging to either distribution.
        """
        if pair_type not in ['pos', 'neg']:
            raise ValueError('A pair type is either "pos" or "neg".')
        return self.prob[pair_type]['gaussian'].pdf(diff_value) * \
            self.prob[pair_type]['prior']

    def evaluate(self, **kwargs):
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
