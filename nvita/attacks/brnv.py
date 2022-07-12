import torch
import numpy as np

from nvita.models.train import predict

class BRNV:
    """
    Baseline Random Sign Method for time series forecasting
    """

    def __init__(self, n, eps, targeted=False) -> None:
        self.n = n
        self.eps = eps
        self.targeted = targeted

    def attack(self, X, n, window_range, seed):
        """
        Randomly select n values to attack
        """
        if X.shape[0] != 1:
            print("Warning, only one window should be inputted to BRNV")
        size_r = X.shape[1] 
        size_c = X.shape[2]
        result = []
        np.random.seed(seed)
        for _ in range(n):
            result.append(np.random.randint(size_r))
            result.append(np.random.randint(size_c))
            if np.random.randint(1):
                result.append(self.eps)
            else:
                result.append((-1) * self.eps)
        X_adv = add_random_perturbation(result, X, window_range)
        return X_adv, result

    def __str__(self) -> str:
        
        if self.targeted:
        
            return "Targeted B" + str(self.n) + "VITA"

        else:

            return "Non-targeted B" + str(self.n) + "VITA"

def add_random_perturbation(eta, x, window_range):
    # Add perturbation based on BRNV result

    X_adv = torch.clone(x).detach()
    for j in range(len(eta)//3):
        row = int(eta[j*3])
        column = int(eta[j*3+1])
        amount = eta[j*3+2]
        X_adv[0,row,column] = X_adv[0,row,column] + amount * window_range[column]
    return X_adv

