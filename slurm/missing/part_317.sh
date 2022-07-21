#!/bin/bash
#SBATSH --job-name=nvita_missing
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

python experiments/step4_attack_non_target.py -d NZTemp -m LSTM -a NVITA -s 789789 -e 0.15 -n 5
python experiments/step4_attack_non_target.py -d NZTemp -m LSTM -a BIM -s 789789 -e 0.2 -n 1
python experiments/step4_attack_non_target.py -d NZTemp -m LSTM -a BIM -s 789789 -e 0.2 -n 1
python experiments/step4_attack_non_target.py -d NZTemp -m LSTM -a BRNV -s 789789 -e 0.2 -n 5
python experiments/step4_attack_non_target.py -d NZTemp -m LSTM -a BRNV -s 789789 -e 0.2 -n 5

