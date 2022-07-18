"""
This script checks the "results" folder and searching for incomplete outputs.

Author: Luke Chang (xcha011@aucklanduni.ac.nz)
Date:   18/07/2022
"""
import os
from pathlib import Path

import pandas as pd

PATH_ROOT = Path(os.getcwd()).absolute()

FOLDER_RESULTS = ['non_targeted_results', 'targeted_results']
TARGETS = ['untargeted', 'targeted']
COLUMNS = ['Seed', 'Target', 'Dataset', 'Model', 'Epsilon', 'Attack', 'Path']


def check_csv_completion(path_file, min_line=101):
    """Returns True if the file contains min number of lines.
    Also returns True if the file only contains titles where the algorithm cannot 
    run under the parameters. For example, some attacks do not work on RF.
    """
    with open(path_file) as file:
        n_lines = len(file.readlines())
        return n_lines >= min_line or n_lines == 1


def parse_filename(filename: str):
    """Extracts metadata from filename"""
    pathname, _extension = os.path.splitext(filename)
    arr = pathname.split('_')
    # Handle the situation:
    # df_Oil_seed_1111111_model_GRU_epsilon_0.15_attack_Non_Targeted_3VITA.csv
    attack = arr[9] if len(arr) == 10 else '_'.join(arr[9:])
    return {
        'Seed': arr[3],
        'Dataset': arr[1],
        'Model': arr[5],
        'Epsilon': arr[7],
        'Attack': attack,
    }


if __name__ == '__main__':
    print(PATH_ROOT)

    df = pd.DataFrame(columns=COLUMNS)
    for folder, target in zip(FOLDER_RESULTS, TARGETS):
        path_results = os.path.join(PATH_ROOT, 'results', folder)
        for root, dirs, files in os.walk(path_results):
            for file in files:
                path_csv = os.path.join(root, file)

                if not check_csv_completion(path_csv):
                    data = parse_filename(file)
                    data['Target'] = target
                    data['Path'] = path_csv
                    df_row = pd.DataFrame([data])
                    df = pd.concat([df, df_row])
    df = df.sort_values(['Seed', 'Target', 'Epsilon', 'Model', 'Attack'])
    print(df)
    df.to_csv(os.path.join(PATH_ROOT, 'results', 'incomplete_results.csv'), index=False)
