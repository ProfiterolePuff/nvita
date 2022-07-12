import numpy as np
import torch
from scipy.optimize import differential_evolution

from nvita.attacks.utils import get_pop_size_for_nvita
from nvita.models.train import predict

class FULLVITA:
    """
    Time-series fullVITA.
    """

    def __init__(self, eps, model, targeted=False) -> None:
        self.eps = eps
        self.model = model
        self.targeted = targeted

    def attack(self, X, target, window_range, maxiter=200, popsize=None, tol=0.01, seed=None):
        """fullVITA.

        Args:
            X: 
                A pytorch tensor which is original input time series with one window
            n: 
                Number of values allowed to be attacked. An interger larger than 1.
            target: 
                Target tuple for targeted attack, true y for non-targeted attack
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

        if popsize == None:
            popsize = get_pop_size_for_nvita(X.shape, len(bounds))

        if self.targeted:

            X_adv_de = differential_evolution(absolute_error_with_target_for_fullvita, bounds, args=(X, target, self.model, window_range), maxiter=maxiter, popsize=popsize, tol=tol, polish=False, seed=seed)

        else:

            X_adv_de = differential_evolution(negative_mse_with_nn_for_fullvita, bounds, args=(X, target, self.model, window_range), maxiter=maxiter, popsize=popsize, tol=tol, polish=False, seed=seed)

        X_adv = X + torch.Tensor(X_adv_de.x).reshape(X.shape) * window_range  

        return X_adv, X_adv_de

    def __str__(self) -> str:
        
        if self.targeted:
        
            return "Targeted_fullVITA"

        else:

            return "Non-targeted_fullVITA"

def absolute_error_with_target_for_fullvita(fullvita_eta, X, target, model, window_range):
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

    return abs(target[1] - predict(model, X_adv).item())

def negative_mse_with_nn_for_fullvita(fullvita_eta, X, y, model, window_range):
    X_adv = X + torch.Tensor(fullvita_eta).reshape(X.shape) * window_range
    y_pred = predict(model, X_adv)
    return -np.sum(((y_pred.detach().numpy().reshape(-1) - y.detach().numpy().reshape(-1))**2)/len(y))
