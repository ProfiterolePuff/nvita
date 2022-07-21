#!/bin/bash
#SBATSH --job-name=nvita_missing
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

python experiments/step5_attack_target.py -d NZTemp -m GRU -a BRS -s 2210 -e 0.1 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m GRU -a FGSM -s 2210 -e 0.1 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m GRU -a FULLVITA -s 2210 -e 0.1 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m GRU -a NOATTACK -s 2210 -e 0.1 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m GRU -a NVITA -s 2210 -e 0.1 -n 1 -t Negative

