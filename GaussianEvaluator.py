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

    def __init__(self, data=None):
        # Superclass constructor sets the datasets, if given.
        super(GaussianEvaluator, self).__init__(self, data)

        # Fitting Gaussian distributions

        positive_diffs = []
        negative_diffs = []
        # Going through every pair of data points.
        for first, second in combinations(range(self._x.shape[0])):
            diff = np.abs(self._x[first, :] - self._x[second, :])
            if self._y[first] == self._y[second]: # A positive (same-class) pair.
                positive_diffs.append(diff)
            else: # A negative (different class) pair.
                negative_diffs.append(diff)

        # Fitting two Gaussians for positive and negative pairs.
        self.positive_gaussian = multivariate_normal(
            np.mean(positive_diffs, axis=0),
            np.cov(positive_diffs, rowvar=0) # Covariance between dimensions.
        )
        # P(pos) = N(pos)/(N(pos) + N(neg))
        self.positive_prior = float(len(positive_diffs)) / \
            (len(positive_diffs) + len(negative_diffs))
        self.negative_gaussian = multivariate_normal(
            np.mean(negative_diffs, axis=0),
            np.cov(negative_diffs, rowvar=0)
        )
        # P(neg) = 1 - P(pos) = N(neg)/(N(pos) + N(neg))
        self.negative_prior = 1.0 - self.positive_prior

    def belong_prob(self, diff_value, pair_type):
        """
        Calculates probability of an absolute difference value belonging to a
        pair type, positive or negative.
        args:
            - diff_value: (ndarray) the absolute difference between two data points.
            - pair_type: ('pos'/'neg') prob of belonging to corresponding Gaussians.
        Returns:
            - (float in [0.0, 1.0]) The probability of the difference value
            belonging to either distribution.
        """
        if pair_type == 'pos':
            return self.positive_gaussian.pdf(diff_value) * self.positive_prior
        elif pair_type == 'neg':
            return self.negative_gaussian.pdf(diff_value) * self.negative_prior
        else:
            raise ValueError('A pair type is either "pos" or "neg".')

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
