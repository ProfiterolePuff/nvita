import random

import numpy as np
import torch

from nvita.models.train import predict

class BRS:
    """
    Baseline Random Sign Method for time series forecasting
    """

    def __init__(self, eps, targeted=False) -> None:
        self.eps = eps
        self.targeted = targeted

    def attack(self, X, window_range, seed):
        """
        Rnadomly select -1 or 1 as the sign of the value
        """
        random.seed(seed)
        X_adv = torch.tensor(np.array([[(-1)**random.randint(0,1) for i in range(X.shape[2])] for j in range(X.shape[1])])).reshape(X.shape)*self.eps*window_range
        return X_adv

    def __str__(self) -> str:
        
        if self.targeted:
        
            return "Targeted BRS"

        else:

            return "Non-targeted BRS"
