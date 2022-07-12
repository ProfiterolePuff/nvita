import torch
import numpy as np

from nvita.models.data import SplittedTSData
from nvita.models.utils import load_model
import nvita.models.train as mt

from nvita.utils import open_json

from nvita.attacks.brs import BRS
from nvita.attacks.brnv import BRNV
from nvita.attacks.fgsm import FGSMTSF
from nvita.attacks.bim import BIMTSF
from nvita.attacks.nvita import NVITA, add_perturbation
from nvita.attacks.fullvita import FULLVITA
from nvita.attacks.utils import create_empty_result_csv_file, append_result_to_csv_file


import time
import argparse
import os
from pathlib import Path

def run_exp(df_name, seed, model, attack, epsilon, n, target):
    path_root = Path(os.getcwd()).parent.absolute()
    path_out_dir = os.path.join(path_root, "results", "targeted_results") 
    path_out_file = os.path.join(path_out_dir, "df_"+df_name+"_seed_"+str(seed)+"_model_"+model+"_epsilon_"+str(epsilon)+"_attack_"+str(attack)+"_target_"+target+".csv")
    ci_levels = [50, 60, 70, 80, 90, 95, 99]
    # all CI levels to test

    first_result_line_list = ["df", "Seed", "Model", "Epsilon", "Targeted", "Target Direction", "Test Index", "Attack Name", "True y", "Target Value", "Original y Pred", "Attacked AE", "Original AE", "Max Per", "Sum Per", "Cost Time"] 

    for ci_level in ci_levels:
        first_result_line_list.append("Value of " + target + str(ci_level))
        first_result_line_list.append("Success in" + target + str(ci_level))

    first_result_line_list += ["Window Range", "Adv Example" ]

    create_empty_result_csv_file(path_out_file, first_result_line_list)
    # Create empty csv file with the column names

    s_data = SplittedTSData()
    s_data = s_data.load_splitted_data(path_root, df_name, seed)
    # load data
    m = load_model(path_root, df_name, seed, model)
    # load model
    tar_val = 0.1
    # set up target value offset

    if model == "RF" and (attack == "FGSM" or attack == "BIM"):
        raise Exception("RandomFOrest can not be attacked by " + attack + " !")
    ["NoAttack", "BRS", "BnRV", "FGSM", "BIM", "nVITA", "fullVITA"],
    
    for test_ind in range(s_data.X_test.shape[0] - 90):
    #for test_ind in range(s_data.X_test.shape[0]):
        X_current = torch.reshape(s_data.X_test[test_ind], s_data.single_X_shape)
        ground_truth_y = torch.reshape(s_data.y_test[test_ind], s_data.single_y_shape)
        window_range = s_data.window_ranges[test_ind]
        mean_pred, std_pred = mt.get_mean_and_std_pred(m, X_current)
        up, lo = mt.get_ci(mean_pred, std_pred, ci_levels[-1])
        if target == "Positive":
            attack_goal = up + tar_val
        elif target == "Negative":
            attack_goal = lo + tar_val
        else:
            raise Exception("target: " + target + "is invalid!")
        attack_goal_tup = (test_ind, attack_goal)

        current_time = time.time()
        if attack == "NOATTACK":
            X_adv = X_current
            att = attack
        elif attack == "BRS":
            att = BRS(epsilon, targeted=True)
            X_adv = att.attack(X_current, window_range, seed+test_ind*2)
        elif attack == "BRNV":
            att = BRNV(n, epsilon, targeted=True)
            X_adv, _ = att.attack(X_current, n, window_range, seed+test_ind*2)
        elif attack == "FGSM": 
            att = FGSMTSF(epsilon, m, loss_type = "MSE", targeted=True)
            X_adv = att.attack(X_current, attack_goal_tup, window_range)
        elif attack == "BIM":
            steps = 200
            alpha = epsilon/steps
            att = BIMTSF(epsilon, alpha, steps, m, loss_type = "MSE", targeted=True)
            X_adv = att.attack(X_current, attack_goal_tup, window_range)
        elif attack == "NVITA":
            att = NVITA(n, epsilon, m, targeted=True)
            X_adv, _ = att.attack(X_current, attack_goal_tup, window_range, seed=seed)
        elif attack == "FULLVITA":
            att = FULLVITA(epsilon, m, targeted=True)
            X_adv, _ = att.attack(X_current, attack_goal_tup, window_range, seed=seed)

        cost_time = time.time() - current_time
        original_y_pred = mt.adv_predict(m, X_current)
        adv_y_pred = mt.adv_predict(m, X_adv)
        original_ae = np.absolute(original_y_pred - attack_goal)
        attacked_ae = np.absolute(adv_y_pred - attack_goal)
        eta = X_adv - X_current
        sum_per = torch.sum(torch.abs(eta)).item()
        max_per = torch.max(torch.abs(eta)).item()
        
        result = [df_name, seed, model, epsilon, "True", target, test_ind, str(att), str(ground_truth_y.item()), str(attack_goal), original_y_pred, attacked_ae, original_ae, max_per, sum_per, cost_time]

        for ci_level in ci_levels:
            up, lo = mt.get_ci(mean_pred, std_pred, ci_level)
            success = 0
            if target == "Positive":
                result.append(up)
                if adv_y_pred > up:
                    success = 1
            elif target == "Negative":
                result.append(lo)
                if adv_y_pred < lo:
                    success = 1
            else:
                raise Exception("target: " + target + "is invalid!")
            
            result.append(success)
        
        result += [str(window_range.tolist()).replace(",", ";"), str(X_adv.tolist()).replace(",", ";")]
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
    parser.add_argument("-t", "--target", type=str, required=True, help="The target , should be" + str(my_metadata["targets"]) + " ;")
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
    if str(args.target) not in my_metadata["targets"]:
        raise Exception("Inputted target " + str(args.target) + " is not in " + str(my_metadata["targets"]))

    run_exp(args.dfname, args.seed, args.model, args.attack, args.epsilon, args.n, args.target)