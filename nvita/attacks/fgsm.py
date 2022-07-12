import torch

from nvita.models.train import predict

class FGSMTSF:
    """
    FGSM for time series forecasting
    """

    def __init__(self, eps, model, loss_type = "MSE", targeted=False) -> None:
        self.eps = eps
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
        y_pred = predict(self.model, X_adv)

        if self.targeted:

            target_value = torch.tensor(target[1]).reshape(y_pred.shape).float()
            loss = self.criterion(y_pred, target_value)
            
        else:

            loss = self.criterion(y_pred, target)
            
        self.model.zero_grad()
        loss.backward()
        data_grad = X_adv.grad.data
        sign_data_grad = data_grad.sign()

        if self.targeted:

            X_adv = X_adv - self.eps * sign_data_grad * window_range

        else:

            X_adv = X_adv + self.eps * sign_data_grad * window_range

        return X_adv

    def __str__(self) -> str:
        
        if self.targeted:
        
            return "Targeted FGSM"

        else:

            return "Non-targeted FGSM"