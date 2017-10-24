"""
This module contains the CycleExamples class, which finds hard examples
according to the paper at:
http://faculty.ucmerced.edu/mhyang/papers/eccv16_feature_learning_supp.pdf
"""

from HardExamples import HardExamples

class CycleExamples(HardExamples):
    """
    This class finds hard positive and negatives based on the method described
    in:
    http://faculty.ucmerced.edu/mhyang/papers/eccv16_feature_learning_supp.pdf
    """
    def __init__(self, data):
        super(CycleExamples, self).__init__(self, data)

    def find_positive(self, k):
        pass

    def find_negative(self, k):
        pass
