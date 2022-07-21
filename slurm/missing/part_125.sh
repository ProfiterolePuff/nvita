#!/bin/bash
#SBATSH --job-name=nvita_missing
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

python experiments/step5_attack_target.py -d NZTemp -m GRU -a BRS -s 789789 -e 0.2 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m GRU -a FGSM -s 789789 -e 0.2 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m GRU -a FULLVITA -s 789789 -e 0.2 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m GRU -a NOATTACK -s 789789 -e 0.2 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m GRU -a NVITA -s 789789 -e 0.2 -n 1 -t Negative

