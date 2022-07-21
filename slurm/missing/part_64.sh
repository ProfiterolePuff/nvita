#!/bin/bash
#SBATSH --job-name=nvita_missing
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

python experiments/step5_attack_target.py -d NZTemp -m LSTM -a NVITA -s 9999 -e 0.2 -n 3 -t Positive
python experiments/step5_attack_target.py -d NZTemp -m LSTM -a NVITA -s 9999 -e 0.2 -n 5 -t Positive
python experiments/step5_attack_target.py -d NZTemp -m RF -a BRNV -s 9999 -e 0.1 -n 5 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m RF -a BRS -s 9999 -e 0.1 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m RF -a FULLVITA -s 9999 -e 0.1 -n 1 -t Negative

