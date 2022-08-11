#!/bin/bash
#SBATSH --job-name=nvita_neg_tar
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=6

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

python experiments/step5_attack_target.py -d Electricity -m CNN -a BRNV -s 2210 -e 0.15 -n 5 -t Negative
python experiments/step5_attack_target.py -d Electricity -m CNN -a BRNV -s 2210 -e 0.2 -n 5 -t Negative
python experiments/step5_attack_target.py -d Electricity -m CNN -a FGSM -s 2210 -e 0.05 -n 1 -t Negative
python experiments/step5_attack_target.py -d Electricity -m CNN -a FGSM -s 2210 -e 0.1 -n 1 -t Negative
python experiments/step5_attack_target.py -d Electricity -m CNN -a FGSM -s 2210 -e 0.15 -n 1 -t Negative
