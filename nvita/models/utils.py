import torch

import os
from pathlib import Path

def save_pytorch_model(model, path_root, df_name, seed):
    path_save = os.path.join(path_root, "results", "saved_models", "df_"+df_name+"_seed_"+str(seed)+"_model_"+str(model)+".pkl")   
    torch.save(model.state_dict(), path_save)