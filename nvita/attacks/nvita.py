import numpy as np
import torch
from scipy.optimize import differential_evolution

from nvita.models.train import adv_predict

class NVITA:
    """
    Time-series nVITA.
    """

    def __init__(self, n, eps, model, targeted=False) -> None:
        self.n = n
        self.eps = eps
        self.model = model
        self.targeted = targeted

    def attack(self, X, target, window_range, maxiter=1000, popsize=15, tol=0.01, seed=None):
        """Targeted NVITA attack.

        Args:
            X: 
                A pytorch tensor which is original input time series with one window
            n: 
                Number of values allowed to be attacked. An interger larger than 1.
            target: 
                Target tuple 
            model: 
                A pytorch TSF model which will be attacked
            window_range: 
                A list reprsents a single window range corresponds to this particular X (window)
            seed:
                A int to make the output of nvita bacome reproducible

        Returns:
            X_adv_de: Adversarial example genereated by DE for NVITA

        """

        bounds = [(0, X.shape[1]), (0, X.shape[2]), (-self.eps, self.eps)] * self.n

        if self.targeted:
            
            X_adv_de = differential_evolution(self.absolute_error_with_target, bounds, args=(X, target, self.model, window_range), maxiter=maxiter, popsize=popsize, tol=tol)
        else:
            
            X_adv_de = differential_evolution(self.negative_mse_with_nn, bounds, args=(X, target, self.model, window_range), maxiter=maxiter, popsize=popsize, tol=tol, seed=seed)
        return X_adv_de


    def add_perturbation(self, nvita_eta, X, window_range):
        """
        Generate crafted adversarial example by adding perturbation crafted by nvita on the original time series 
        
        Args:
            nvita_eta: 
                A list reprsents the perturbation added by nvita
            X: 
                A pytorch tensor which is original input time series with one window
            window_range: 
                A list reprsents a single window range corresponds to this particular X (window)

        Returns:
            A pytorch tensor which is the crafted adversarial example
        """
        X_adv = torch.clone(X).detach()

        for attack_value_index in range(len(nvita_eta)//3):
            row_ind = int(nvita_eta[attack_value_index * 3])
            col_ind = int(nvita_eta[attack_value_index * 3 + 1])
            amount = nvita_eta[attack_value_index * 3 + 2]

            if X_adv[0, row_ind, col_ind] == X[0, row_ind, col_ind]:
                # If the value is not attacked, it is allowed to be attacked
                X_adv[0, row_ind, col_ind] = X_adv[0, row_ind, col_ind] + amount * window_range[col_ind]

        return X_adv


    def negative_mse_with_nn(self, eta, X, y, window_range):
        X_adv = self.add_perturbation(eta, X, window_range)
        y_pred = adv_predict(self.model, X_adv)
        return -np.sum(((y_pred.detach().numpy().reshape(-1) - y.detach().numpy().reshape(-1))**2)/len(y))


    def absolute_error_with_target(self, nvita_eta, X, target, window_range):
        """
        Calculate abosolute error between the attacked model perdiction and the target
        Used as fitness function for targeted_nvita
        Assuming the model will only output exactly one prediction and we only have one target

        Args:
            nvita_eta: 
                A list reprsents the perturbation added by nvita
            X: 
                A pytorch tensor which is original input time series with one window
            target: 
                Target tuple 
            model: 
                A pytorch TSF model which will be attacked
            window_range: 
                A list reprsents a single window range corresponds to this particular X (window)

        Returns:
            A float which is the abosolute error between the attacked model perdiction and the target
        """

        X_adv = self.add_perturbation(nvita_eta, X, window_range)

        return abs(target[1] - adv_predict(self.model, X_adv).item())

    def __str__(self) -> str:
        
        if self.targeted:
        
            return "Targeted " + str(self.n) + "VITA"

        else:

            return "Non-targeted" + str(self.n) + "VITA"
