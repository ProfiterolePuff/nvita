import numpy as np
import torch
from scipy.optimize import differential_evolution

from nvita.models.train import adv_predict

class FULLVITA:
    """
    Time-series nVITA.
    """

    def __init__(self, eps, model, targeted=False) -> None:
        self.eps = eps
        self.model = model
        self.targeted = targeted

    def attack(self, X, target, window_range, maxiter=1000, popsize=15, tol=0.01, seed=None):
        """Targeted FULLVITA attack.

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
            X_adv_de: Adversarial example genereated by DE for FULLVITA
        """
        bounds = [(-self.eps, self.eps)] * X.shape[1] * X.shape[2]

        if self.targeted:

            X_adv_de = differential_evolution(self.absolute_error_with_target_for_fullvita, bounds, args=(X, target, model, window_range), maxiter=maxiter, popsize=popsize, tol=tol)
        else:

            X_adv_de = differential_evolution(self.negative_mse_with_nn_for_fullvita, bounds, args=(X, y, model, window_range), maxiter=maxiter, popsize=popsize, tol=tol)

        return X_adv_de

    def absolute_error_with_target_for_fullvita(self, fullvita_eta, X, target, model, window_range):
        """
        Calculate abosolute error between the attacked model perdiction and the target
        Used as fitness function for targeted_fullvita
        Assuming the model will only output exactly one prediction and we only have one target

        Args:
            fullvita_eta: 
                A list reprsents the perturbation added by fullvita
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
        X_adv = X + torch.Tensor(fullvita_eta).reshape(X.shape) * window_range

        return abs(target[1] - adv_predict(self.model, X_adv).item())

    def negative_mse_with_nn_for_fullvita(self, fullvita_eta, X, y, model, window_range):
        X_adv = X + torch.Tensor(fullvita_eta).reshape(X.shape) * window_range
        y_pred = adv_predict(self.model, X_adv)
        return -np.sum(((y_pred.detach().numpy().reshape(-1) - y.detach().numpy().reshape(-1))**2)/len(y))


