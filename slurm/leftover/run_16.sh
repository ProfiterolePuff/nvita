#!/bin/bash
#SBATSH --job-name=nvita_leftover
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

python experiments/step4_attack_non_target.py -d Oil -m GRU -a NVITA -s 1111111 -e 0.15 -n 1

