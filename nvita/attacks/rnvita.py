import numpy as np
import torch
from scipy.optimize import differential_evolution

from nvita.attacks.utils import get_pop_size_for_nvita
from nvita.models.train import predict

class RNVITA:
    """
    Time-series nVITA with restraints.
    """

    def __init__(self, n, eps, model, targeted=False) -> None:
        self.n = n
        self.eps = eps
        self.model = model
        self.targeted = targeted

    def attack(self, X, target, window_range, feature_restraints, maxiter=200, popsize=None, tol=0.01, seed=None):
        """ RNVITA attack.

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
            feature_restraints:
                A list reprsents the feature (column) indices that do not allow to be attacked. 
                The feature index in the feature_restraints must be the head(0) or tail(X.shape[1]) or at least connected with the head or tail
            seed:
                A int to make the output of nvita bacome reproducible

        Returns:
            X_adv_de: Adversarial example genereated by DE for NVITA

        """
        rinds = self.get_feature_cut_indices(feature_restraints,  X.shape[2])
        remnant = X[:, :, rinds[0]:rinds[1]]
        head = X[:, :, :rinds[0]]
        tail = X[:, :, rinds[1]:]

        bounds = [(0, remnant.shape[1]), (0, remnant.shape[2]), (-self.eps, self.eps)] * self.n

        if popsize == None:
            popsize = get_pop_size_for_nvita(X.shape, len(bounds))

        if self.targeted:

            X_adv_de = differential_evolution(absolute_error_with_target, bounds, args=(remnant, head, tail, target, self.model, window_range), maxiter=maxiter, popsize=popsize, tol=tol, polish=False, seed=seed)
        else:

            X_adv_de = differential_evolution(negative_mse_with_true_y, bounds, args=(remnant, head, tail, target, self.model, window_range), maxiter=maxiter, popsize=popsize, tol=tol, polish=False, seed=seed)

        X_adv = add_perturbation_with_restraints(X_adv_de.x, remnant, head, tail, window_range)
        return X_adv, X_adv_de

    def __str__(self) -> str:
        
        if self.targeted:
        
            return "Targeted_" + str(self.n) + "VITA"

        else:

            return "Non-targeted_" + str(self.n) + "VITA"

    def get_feature_cut_indices(self, feature_restraints, tail_ind):
        """ 
        Get the indices for removing the feature (column) indices that do not allow to be attacked. 

        """
        if feature_restraints == []:
            # Without restraints
            return(0, tail_ind)

        head = 0
        count = 0
        prev = None

        for i in feature_restraints:

            if i != head+count:

                if prev is None:
                    return(0, i - 1)
                else:
                    return(prev + 1, i - 1)

            count += 1
            prev = i
        
        return(feature_restraints[-1] + 1, tail_ind)

def add_perturbation_with_restraints(nvita_eta, remnant, head, tail, window_range):
    """
    Generate crafted adversarial example by adding perturbation crafted by nvita on the original time series 
    
    Args:
        nvita_eta: 
            A list reprsents the perturbation added by nvita
        remnant, head, tail: 
            Three parts are cut by the pytorch tensor which is original input time series with one window 
        window_range: 
            A list reprsents a single window range corresponds to this particular X (window)

    Returns:
        A pytorch tensor which is the crafted adversarial example
    """
    remnant_adv = torch.clone(remnant).detach()

    for attack_value_index in range(len(nvita_eta)//3):
        row_ind = int(nvita_eta[attack_value_index * 3])
        col_ind = int(nvita_eta[attack_value_index * 3 + 1])
        amount = nvita_eta[attack_value_index * 3 + 2]

        remnant_adv[0, row_ind, col_ind] = remnant[0, row_ind, col_ind] + amount * window_range[col_ind]
        # New perturbation overrides the old if they attacked the same value
    X_adv = torch.cat((head, remnant_adv, tail), 2)
    return X_adv


def negative_mse_with_true_y(eta, remnant, head, tail, y, model, window_range):
    X_adv = add_perturbation_with_restraints(eta, remnant, head, tail, window_range)
    y_pred = predict(model, X_adv)
    return -np.sum(((y_pred.detach().numpy().reshape(-1) - y.detach().numpy().reshape(-1))**2)/len(y))


def absolute_error_with_target(nvita_eta, remnant, head, tail, target, model, window_range):
    """
    Calculate abosolute error between the attacked model perdiction and the target
    Used as fitness function for targeted_nvita
    Assuming the model will only output exactly one prediction and we only have one target

    Args:
        nvita_eta: 
            A list reprsents the perturbation added by nvita
        remnant, head, tail: 
            Three parts are cut by the pytorch tensor which is original input time series with one window 
        target: 
            Target tuple 
        model: 
            A pytorch TSF model which will be attacked
        window_range: 
            A list reprsents a single window range corresponds to this particular X (window)

    Returns:
        A float which is the abosolute error between the attacked model perdiction and the target
    """

    X_adv = add_perturbation_with_restraints(nvita_eta, remnant, head, tail, window_range)

    return abs(target[1] - predict(model, X_adv).item())
