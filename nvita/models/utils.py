import torch

import os
import pickle

from nvita.utils import check_file_existence

def save_model(model, path_root, df_name, seed):
    path_save = os.path.join(path_root, "results", "saved_models", "df_"+df_name+"_seed_"+str(seed)+"_model_"+str(model)+".pkl")  
    if not check_file_existence(path_save):
        with open(path_save, 'wb') as out:
            if str(model) == "RF":
                #save sklearn random forest model
                pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)
            else:
                #save pytorch model
                torch.save(model, path_save)

def load_model(path_root, df_name, seed, model_name):
    """
    Load model with given df name, seed and model name, return the model object
    """
    path_load = os.path.join(path_root, "results", "saved_models", "df_"+df_name+"_seed_"+str(seed)+"_model_"+model_name+".pkl")  
    with open(path_load, 'rb') as inp:
        if model_name == "RF":
            result = pickle.load(inp)
        else:
            result = torch.load(path_load)
            result.eval()
    return result
