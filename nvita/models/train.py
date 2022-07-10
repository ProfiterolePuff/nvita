
import numpy as np
import torch

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
    if str(model) == "RF":
        result = model.rf_pytorch_predict(X)
    else:
        model.eval()
        result = np.ones(sample)
        for s in range(sample):
            result[s] = model(X).item()

    return np.mean(result), result


