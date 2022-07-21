#!/bin/bash
#SBATSH --job-name=nvita_missing
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

python experiments/step5_attack_target.py -d NZTemp -m LSTM -a BIM -s 1111111 -e 0.2 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m LSTM -a BRNV -s 1111111 -e 0.2 -n 5 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m LSTM -a BRS -s 1111111 -e 0.2 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m LSTM -a FGSM -s 1111111 -e 0.2 -n 1 -t Negative
python experiments/step5_attack_target.py -d NZTemp -m LSTM -a FULLVITA -s 1111111 -e 0.2 -n 1 -t Negative

