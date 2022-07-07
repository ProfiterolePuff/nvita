# targeted fullvita

def absolute_error_with_target_for_fullvita(fullvita_eta, x, target, model, window_range):
    """
    Calculate abosolute error between the attacked model perdiction and the target
    Used as fitness function for targeted_fullvita
    Assuming the model will only output exactly one prediction and we only have one target

    Args:
        fullvita_eta: 
            A list reprsents the perturbation added by fullvita
        x: 
            A pytorch tensor which is original input time series with one window
        target: 
            Target tuple 
        model: 
            A pytorch TSF model which will be attacked
        window_range: 
            A list reprsents a single window range corresponds to this particular x (window)

    Returns:
        A float which is the abosolute error between the attacked model perdiction and the target
    """
    x_dash = x + torch.Tensor(fullvita_eta).reshape(x.shape) * window_range

    return abs(target[1] - model(x_dash).item())

def targeted_fullvita(x, target, epsilon , model, window_range, maxiter=200, popsize=15, tol=0.01, seed = None):
    """Targeted FULLVITA attack.

    Args:
        x: 
            A pytorch tensor which is original input time series with one window
        n: 
            Number of values allowed to be attacked. An interger larger than 1.
        target: 
            Target tuple 
        model: 
            A pytorch TSF model which will be attacked
        window_range: 
            A list reprsents a single window range corresponds to this particular x (window)
        seed:
            A int to make the output of nvita bacome reproducible

    Returns:
        x_dash_de: Adversarial example genereated by DE for FULLVITA
    """
    bounds = [(-epsilon, epsilon)] * x.shape[1] * x.shape[2]
    x_dash_de = differential_evolution(absolute_error_with_target_for_fullvita, bounds, args=(x, target, model, window_range), maxiter=maxiter, popsize=popsize, tol=tol)

    return x_dash_de

# Non-targeted fullvita

def negative_mse_with_nn_for_fullvita(fullvita_eta, x, y, model, window_range):
    x_dash = x + torch.Tensor(fullvita_eta).reshape(x.shape) * window_range
    y_pred = model(x_dash)
    return -np.sum(((y_pred.detach().numpy().reshape(-1) - y.detach().numpy().reshape(-1))**2)/len(y))

def non_targeted_fullvita(x, y, n, epsilon, model, window_range, maxiter=1000, popsize=15, tol=0.01, seed=None):
    bounds = [(-epsilon, epsilon)] * x.shape[1] * x.shape[2]
    x_dash_de = differential_evolution(negative_mse_with_nn_for_fullvita, bounds, args=(x, y, model, window_range), maxiter=maxiter, popsize=popsize, tol=tol)
    return x_dash_de