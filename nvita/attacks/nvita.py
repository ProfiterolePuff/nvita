from cmath import inf
import numpy as np

class NVITA:
    """
    Time-series nVITA.
    """

    def __init__(self, targeted=False, n=1) -> None:
        if n == 1:
            # 1vita
            pass
        elif n == np.inf:
            # full vita
            pass
        pass

    def generate(self, X, y=None):
        return X

