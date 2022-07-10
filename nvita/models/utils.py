import torch

import os

from nvita.utils import check_file_existence

def save_pytorch_model(model, path_root, df_name, seed):
    path_save = os.path.join(path_root, "results", "saved_models", "df_"+df_name+"_seed_"+str(seed)+"_model_"+str(model)+".pkl")  
    if not check_file_existence(path_save):
        torch.save(model.state_dict(), path_save)
