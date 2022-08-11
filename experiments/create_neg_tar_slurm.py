"""
This script creates Slurm script for Negative Targeted experiments.

Author: Luke Chang (xcha011@aucklanduni.ac.nz)
Date:   11/09/2022
"""

from importlib.metadata import metadata
import os
from pathlib import Path
import json

import pandas as pd

from nvita.utils import create_dir

PATH_ROOT = Path(os.getcwd()).absolute()
PATH_RESULTS = os.path.join(PATH_ROOT, 'results')
PATH_OUTPUT = os.path.join(PATH_ROOT, 'new_slurm')

SLURM_HEADER = """#!/bin/bash
#SBATSH --job-name=nvita_neg_tar
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=6

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate
"""


def script_builder(dataset, model, attack, seed, eps, n, direction='Negative'):
    script = f'python experiments/step5_attack_target.py -d {dataset} -m {model} -a {attack} -s {seed} -e {eps} -n {n} -t {direction}'
    return script


def generate(path_json=os.path.join(PATH_ROOT, 'experiments', 'metadata.json')):
    with open(path_json) as file:
        metadata = json.load(file)
    data = metadata['data']
    seeds = metadata['seeds']
    models = metadata['models']
    attacks = metadata['attacks']
    n_values = metadata['n_values']
    epsilons = metadata['epsilons']

    scripts = []
    for d in data:
        for s in seeds:
            for m in models:
                for att in attacks:
                    for eps in epsilons:
                        if att == 'NVITA':
                            for n in n_values:
                                scripts.append(script_builder(d, m, att, s, eps, n))
                        elif att == 'BRNV':
                            scripts.append(script_builder(d, m, att, s, eps, n=5))
                        else:
                            if (att == 'FGSM' and m == 'RF') or (att == 'BIM' and m == 'RF'):
                                continue
                            scripts.append(script_builder(d, m, att, s, eps, n=1))
    print(*scripts[:5], sep='\n')
    print(f'Total exp: {len(scripts)}')
    return scripts


def save_script(line, output, idx):
    my_script = SLURM_HEADER + '\n' + line + '\n'
    output_name = f'run_{idx}.sh'
    with open(os.path.join(PATH_ROOT, output, output_name), 'w') as file:
        file.writelines(my_script)


def get_content(n, dir):
    content = '''#!/bin/bash

for I in {{1..{}}}; do
    sbatch ./{}/part_$I.sh
done
        
squeue --me
'''.format(n, dir)
    return content


if __name__ == '__main__':
    print('PATH_ROOT:', PATH_ROOT)
    create_dir(PATH_OUTPUT)
    print('PATH_OUTPUT:', PATH_OUTPUT)

    scripts = generate()

    file_count = 1
    for i in range(0, len(scripts), 5):
        start = i
        end = len(scripts) if i + 5 >= len(scripts) else i + 5

        lines = '\n'.join(scripts[start:end])
        save_script(lines, PATH_OUTPUT, file_count)
        file_count += 1

    # Create the script to submit all files
    file_count = file_count - 1
    with open(os.path.join(PATH_OUTPUT, 'run_missing.sh'), 'w') as file:
        file.writelines(get_content(file_count, 'new_slurm'))
    print(f'Created {file_count} slurm scripts in total')
