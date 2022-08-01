#!/bin/bash
#SBATSH --job-name=nvita_missing
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

python experiments/step4_attack_non_target.py -d CNYExch -m RF -a NVITA -s 58361 -e 0.2 -n 5
python experiments/step4_attack_non_target.py -d Oil -m CNN -a BIM -s 58361 -e 0.2 -n 1
python experiments/step4_attack_non_target.py -d Oil -m CNN -a BRNV -s 58361 -e 0.2 -n 5
python experiments/step4_attack_non_target.py -d Oil -m CNN -a BRS -s 58361 -e 0.2 -n 1
python experiments/step4_attack_non_target.py -d Oil -m CNN -a FGSM -s 58361 -e 0.2 -n 1
