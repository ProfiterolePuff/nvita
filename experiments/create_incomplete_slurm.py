"""
This script reads the CSV file generated by `check_results.py` and creates slurm 
script for cluster computing.

Author: Luke Chang (xcha011@aucklanduni.ac.nz)
Date:   18/07/2022
"""
import os
from pathlib import Path

import pandas as pd

PATH_ROOT = Path(os.getcwd()).absolute()
PATH_RESULTS = os.path.join(PATH_ROOT, 'results')

TARGET_SCRIPT = {
    'untargeted': 'step4_attack_non_target.py',
    'targeted': 'step5_attack_target.py',
}
SLURM_HEADER = """#!/bin/bash
#SBATSH --job-name=nvita_leftover
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate
"""


def parse_attack(attack_str: str):
    """Parses attack value, and returns attack_name, n, pos/neg as a tuple."""
    n = 1
    direction = None
    if 'FULLVITA' in attack_str:
        att = 'FULLVITA'
    elif 'NOATTACK' in attack_str:
        att = 'NOATTACK'
    elif 'BIM' in attack_str:
        att = 'BIM'
    elif 'BRS' in attack_str:
        att = 'BRS'
    elif '1VITA' in attack_str:
        att = 'NVITA'
    elif '3VITA' in attack_str:
        att = 'NVITA'
    elif '5VITA' in attack_str:
        att = 'NVITA'
    else:
        if len(attack_str.split('_')) != 1:
            print(f'Cannot handle "{attack_str}".')
        att = attack_str

    _dir = attack_str.split('_')[-1]
    if _dir in ["Positive", "Negative"]:
        direction = _dir
    return att, n, direction


def script_builder(target, dataset, model, attack, seed, eps, n, direction=None):
    script = f'python experiments/{TARGET_SCRIPT[target]} -d {dataset} -m {model} -a {attack} -s {seed} -e {eps} -n {n}'
    if target == 'targeted':
        script += f' -t {direction}'
    script += '\n'
    return script


def create_script(row_data):
    attack, n, direction = parse_attack(row_data['Attack'])
    script = script_builder(
        target=row_data['Target'],
        dataset=row_data['Dataset'],
        model=row_data['Model'],
        attack=attack,
        seed=row_data['Seed'],
        eps=row_data['Epsilon'],
        n=n,
        direction=direction,
    )
    return script


def save_script(line, output_dir, idx):
    my_script = SLURM_HEADER + '\n' + line + '\n'
    output_name = f'run_{idx}.sh'
    with open(os.path.join(PATH_ROOT, output_dir, output_name), 'w') as file:
        file.writelines(my_script)


if __name__ == '__main__':
    print('PATH_ROOT:', PATH_ROOT)

    lines = []
    df = pd.read_csv(os.path.join(PATH_RESULTS, 'incomplete_results.csv'))
    for i in range(df.shape[0]):
        row_data = df.iloc[i]
        script = create_script(row_data)
        lines.append(script)

    for idx, line in enumerate(lines):
        save_script(line, os.path.join('slurm', 'leftover'), idx + 1)

    print(f'Generated {len(lines)} files in total.')
