from sklearn.ensemble import RandomForestRegressor
import numpy as np
import torch

class RF():
    """
    Random Forest Regressor from sklearn
    """
    def __init__(self, n_estimators):
        self.model = RandomForestRegressor(n_estimators)
    
    def fit(self, X, y):
        """
        train the model with pytorch tensor X and y
        """
        self.model.fit([a.reshape(-1) for a in X.detach().numpy()], y.detach().numpy().reshape(-1))

    def rf_pytorch_predict(self, X):
        """
        predict with pytorch tensor X
        """
        result = []
        for X_ind in range(X.shape[0]):
            result.append(self.model.predict([[a.reshape(-1) for a in X.detach().numpy()][X_ind]]))
        result = np.array(result)
        result = torch.from_numpy(result)
        return result

    def __str__(self) -> str:
        return "RF"