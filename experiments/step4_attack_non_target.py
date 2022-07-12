import argparse
import os
import time
from pathlib import Path

import numpy as np
import nvita.models.train as mt
import torch
from nvita.attacks.bim import BIMTSF
from nvita.attacks.brnv import BRNV
from nvita.attacks.brs import BRS
from nvita.attacks.fgsm import FGSMTSF
from nvita.attacks.fullvita import FULLVITA
from nvita.attacks.nvita import NVITA
from nvita.attacks.utils import (append_result_to_csv_file,
                                 check_result_file_path,
                                 create_empty_result_csv_file)
from nvita.models.data import SplittedTSData
from nvita.models.utils import load_model
from nvita.utils import create_dir, open_json


def run_exp(df_name, seed, model, attack, epsilon, n, demo):
    path_root = Path(os.getcwd()).parent.absolute()
    # get root path
    s_data = SplittedTSData()
    s_data = s_data.load_splitted_data(path_root, df_name, seed)
    # load data
    m = load_model(path_root, df_name, seed, model)
    # load model

    if demo == None:
        path_out_dir = os.path.join(path_root, "results", "exp_seed_" + str(seed), "exp_" + df_name , "targeted_results")
    else:
        path_out_dir = os.path.join(path_root, "examples", "exp_seed_" + str(seed), "exp_" + df_name , "targeted_results")
    path_out_dir = os.path.join(path_root, "results", "exp_"+str(seed), "non_targeted_results") 
    create_dir(path_out_dir)
    path_out_file = os.path.join(path_out_dir, "df_"+df_name+"_seed_"+str(seed)+"_model_"+str(m)+"_epsilon_"+str(epsilon)+"_attack_"+str(attack)+".csv")
    path_out_file = check_result_file_path(path_out_file)
    first_result_line_list = ["df", "Seed", "Model", "Epsilon", "Targeted", "Test Index", "Attack Name", "True y", "Original y Pred", "Attacked y Pred", "Attacked AE", "Original AE", "Max Per", "Sum Per", "Cost Time", "Window Range", "Adv Example" ]
    create_empty_result_csv_file(path_out_file, first_result_line_list)
    # Create empty csv file with the column names

    if model == "RF" and (attack == "FGSM" or attack == "BIM"):
        raise Exception("RandomFOrest can not be attacked by " + attack + " !")
    
    if demo == None:
        max_w_size = s_data.X_test.shape[0]
    else:
        max_w_size = demo

    for test_ind in range(max_w_size):
        X_current = torch.reshape(s_data.X_test[test_ind], s_data.single_X_shape)
        ground_truth_y = torch.reshape(s_data.y_test[test_ind], s_data.single_y_shape)
        window_range = s_data.window_ranges[test_ind]

        current_time = time.time()
        if attack == "NOATTACK":
            X_adv = X_current
            att = attack
        elif attack == "BRS":
            att = BRS(epsilon)
            X_adv = att.attack(X_current, window_range, seed+test_ind*2)
        elif attack == "BRNV":
            att = BRNV(n, epsilon)
            X_adv, _ = att.attack(X_current, n, window_range, seed+test_ind*2)
        elif attack == "FGSM": 
            att = FGSMTSF(epsilon, m, loss_type = "MSE")
            X_adv = att.attack(X_current, ground_truth_y, window_range)
        elif attack == "BIM":
            steps = 200
            alpha = epsilon/steps
            att = BIMTSF(epsilon, alpha, steps, m, loss_type = "MSE")
            X_adv = att.attack(X_current, ground_truth_y, window_range)
        elif attack == "NVITA":
            att = NVITA(n, epsilon, m)
            X_adv, _ = att.attack(X_current, ground_truth_y, window_range, seed=seed)
        elif attack == "FULLVITA":
            att = FULLVITA(epsilon, m)
            X_adv, _ = att.attack(X_current, ground_truth_y, window_range, seed=seed)

        cost_time = time.time() - current_time
        original_y_pred = mt.adv_predict(m, X_current)
        original_ae = np.absolute(original_y_pred - ground_truth_y.item())
        adv_y_pred = mt.adv_predict(m, X_adv)
        attacked_ae = np.absolute(adv_y_pred - ground_truth_y.item())
        eta = X_adv - X_current
        sum_per = torch.sum(torch.abs(eta)).item()
        max_per = torch.max(torch.abs(eta)).item()
        
        
        result = [df_name, seed, model, epsilon, "False", test_ind, str(att), str(ground_truth_y.item()), str(adv_y_pred), original_y_pred, attacked_ae, original_ae, max_per, sum_per, cost_time, str(window_range.tolist()).replace(",", ";"), str(X_adv.tolist()).replace(",", ";")]
        append_result_to_csv_file(path_out_file, result)


if __name__ == "__main__":
    path_root = Path(os.getcwd()).parent.absolute()
    my_metadata = open_json(os.path.join(
        path_root, "experiments", "metadata.json"))
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dfname", type=str, required=True, help="The name of the dataset, can be "+ str(my_metadata["data"]) + " ;")
    parser.add_argument("-s", "--seed", type=int, required=True, help="The RNG seed, can be " + str(my_metadata["seeds"]) + " ;")
    parser.add_argument("-m", "--model", type=str, required=True, help="The attacked model name, can be " + str(my_metadata["models"]) + " ;")
    parser.add_argument("-a", "--attack", type=str, required=True, help="The attack name, can be " + str(my_metadata["attacks"]) + " ;")
    parser.add_argument("-e", "--epsilon", type=float, required=True, help="The epsilon values, should be " + str(my_metadata["epsilons"]) + " ;")
    parser.add_argument("-n", "--n", type=int, required=True, help="The n value, used in n BnRV and nVITA;")
    parser.add_argument("--demo", type=int, required=False, help="The demo test interger , should be range from (1, 100) ;")
    args = parser.parse_args()

    if str(args.dfname) not in my_metadata["data"]:
        raise Exception("Inputted data name " + str(args.dfname) + " is not in " + str(my_metadata["data"]))
    if str(args.seed) not in my_metadata["seeds"]:
        raise Exception("Inputted seed " + str(args.seed) + " is not in " + str(my_metadata["seeds"]))
    if str(args.model) not in my_metadata["models"]:
        raise Exception("Inputted model name " + str(args.model) + " is not in " + str(my_metadata["models"]))
    if str(args.attack) not in my_metadata["attacks"]:
        raise Exception("Inputted attack name " + str(args.attack) + " is not in " + str(my_metadata["attacks"]))
    if str(args.epsilon) not in my_metadata["epsilons"]:
        raise Exception("Inputted epsilon " + str(args.epsilon) + " is not in " + str(my_metadata["epsilons"]))

    run_exp(args.dfname, args.seed, args.model, args.attack, args.epsilon, args.n, args.demo)
