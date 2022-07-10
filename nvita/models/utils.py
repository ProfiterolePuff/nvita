import torch

import os
<<<<<<< HEAD

from nvita.utils import check_file_existence

def save_pytorch_model(model, path_root, df_name, seed):
    path_save = os.path.join(path_root, "results", "saved_models", "df_"+df_name+"_seed_"+str(seed)+"_model_"+str(model)+".pkl")  
    if not check_file_existence(path_save):
        torch.save(model.state_dict(), path_save)
=======
from pathlib import Path

def save_pytorch_model(model, path_root, df_name, seed):
    path_save = os.path.join(path_root, "results", "saved_models", "df_"+df_name+"_seed_"+str(seed)+"_model_"+str(model)+".pkl")   
    torch.save(model.state_dict(), path_save)
>>>>>>> c8cf62a66ce8c8a88adaf1822052f4db49dbe8a8
