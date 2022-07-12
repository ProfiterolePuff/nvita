
import numpy as np
import torch
import scipy.stats as st

import time

def train(model, lr, num_epochs, x_train, y_train, print_time = False):
    model.train()
    criterion = torch.nn.MSELoss(reduction = "mean")
    optimiser = torch.optim.Adam(model.parameters(), lr = lr)
    start_time = time.time()
    losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        losses[epoch] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    cost_time = time.time() - start_time

    if print_time:
        print("Training time of " + str(model) + " is: " + str(cost_time) + " Sec.")
        
    return losses 

def evaluate(model, X, y):
    """
    return error array
    """
    if str(model) != "RF":
        model.eval()
    return (predict(model, X) - y).detach().numpy()

def predict(model, X):
    if str(model) == "RF":
        result = model.rf_pytorch_predict(X)
    else:
        model.eval()
        result = model(X)
    return result

def adv_predict(model, X, sample = 100):
    """
    Assume X contains only one window
    if the model is pytorch model, return the mean of 100 times prediction
    if the model is sklearn random forest, return the mean of the forest 
    """
    if str(model) == "RF":
        result = model.rf_pytorch_predict(X)

        return result.item()
        
    else:
        model.eval()
        result = np.ones(sample)
        for s in range(sample):
            result[s] = model(X).item()

        return np.mean(result)

def get_mean_and_std_pred(model, X, sample = 100):
    """
    Assume X contains only one window
    if the model is pytorch model, return the mean of 100 times prediction and the std
    if the model is sklearn random forest, return the mean of the forest and std of the forest
    """
    if str(model) == "RF":

        X_arr = X.detach().numpy().reshape(-1)
        result = np.array([tree.predict(X_arr.reshape(1, X_arr.shape[0])) for tree in model.model.estimators_])

    else:
        model.eval()
        result = np.ones(sample)
        for s in range(sample):
            result[s] = model(X).item()

    return np.mean(result), np.std(result)

def get_ci(mean_pred, std_pred, ci_level):
    """
    get CI values with given mean std and confidence level
    """
    z = st.norm.ppf(ci_level/100)
    upper_q = mean_pred + z * std_pred
    lower_q = mean_pred - z * std_pred
    return upper_q, lower_q














#ci_level
#st.norm.ppf(ci_level)


