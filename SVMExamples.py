"""
This module contains the Exemplar SVM method, which finds hard examples
according to the paper at:
https://www.cs.cmu.edu/~efros/exemplarsvm-iccv11.pdf
"""

from HardExamples import HardExamples

class SVMExamples(HardExamples):
    """
    The class for the Exemplar SVM method for retrieving hard examples in a
    dataset. The paper can be found here:
    https://www.cs.cmu.edu/~efros/exemplarsvm-iccv11.pdf
    """

    def __init__(self, data):
        super(SVMExamples, self).__init__(self, data)

    def find_positive(self, k):
        pass

    def find_negative(self, k):
        pass
