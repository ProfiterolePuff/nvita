import os
from pathlib import Path

from nvita.utils import check_file_existence

def get_pop_size_for_nvita(X_shape, bound_len):
    """
    Get popsize if popsize is not specified in nvita or fullvita
    """
    pop_size = 1
    for shape in X_shape:
        pop_size = pop_size * shape
        # Population Size, calculated in respect of input size
    pop_size_mul = max(1, pop_size//bound_len)
    # Population multiplier, in terms of the size of the perturbation vector x
    return pop_size_mul

def get_csv_line_from_list(line_list):
    """
    Create a comma seperated line without new line character with given list
    """
    line_str = ""
    for value in line_list:
        line_str += str(value)
        line_str += ","
    return line_str

def create_empty_result_csv_file(path_save, first_line_list):

    if not check_file_existence(path_save):

        line = get_csv_line_from_list(first_line_list)

        with open(path_save, "a") as f:
            f.write(line)
            f.write("\n")
    
def append_result_to_csv_file(path_save, line_list):

    line = get_csv_line_from_list(line_list)

    with open(path_save, "a") as f:
            f.write(line)
            f.write("\n")

def check_result_file_path(path_file):
    """
    while the path_file exists, add 1 before .csv in the path_file
    """
    while os.path.exists(path_file):
        path_file = Path(str(path_file)[:-4]+"1.csv")
    return path_file
