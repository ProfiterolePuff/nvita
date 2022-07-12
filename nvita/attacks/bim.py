import torch

from nvita.models.train import predict

class BIMTSF:
    """
    BIM for time series forecasting
    """

    def __init__(self, eps, alpha, steps, model, loss_type = "MSE", targeted=False) -> None:
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.model = model

        if loss_type == "MSE":
            self.criterion = torch.nn.MSELoss(reduction = "mean")
        elif loss_type == "L1":
            self.criterion = torch.nn.L1Loss(reduction = "mean")
        else:
            raise Exception("Loss function" + str(loss_type) + "is not \"MSE\" or \"L1\"")

        self.targeted = targeted

    def attack(self, X, target, window_range):
        """
        For non-targeted attack, target is ground truth y
        For targeted attack, target is (i, g)
        """
        X_adv = torch.clone(X).detach()
        X_adv.requires_grad = True
        min_X_adv = X - self.eps * window_range
        max_X_adv = X + self.eps * window_range
        y_pred = predict(self.model, X_adv)

        if self.targeted:

            target_value = torch.tensor(target[1]).reshape(y_pred.shape).float()

        else:

            loss = self.criterion(y_pred, target)

        for _ in range(self.steps):

            y_pred = predict(self.model, X_adv)
            self.model.zero_grad()

            if self.targeted:

                loss = self.criterion(y_pred, target_value)
            else:

                loss = self.criterion(y_pred, target)
            
            loss.backward()
            data_grad = X_adv.grad.data
            sign_data_grad = data_grad.sign()

            if self.targeted:

                X_adv = torch.clamp(X_adv - self.alpha * sign_data_grad * window_range, min_X_adv, max_X_adv).detach()
                
            else:

                X_adv = torch.clamp(X_adv + self.alpha * sign_data_grad * window_range, min_X_adv, max_X_adv).detach()

            X_adv.requires_grad = True

        return X_adv

    def __str__(self) -> str:
        
        if self.targeted:
        
            return "Targeted BIM"

        else:

            return "Non-targeted BIM"
