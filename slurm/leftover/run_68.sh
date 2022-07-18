#!/bin/bash
#SBATSH --job-name=nvita_leftover
#SBATCH --output=log/log_%%x_%%j_%%a.out
#SBATCH --error=log/log_%%x_%%j_%%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

python experiments/step5_attack_target.py -d Electricity -m CNN -a NOATTACK -s 9999 -e 0.05 -n 1 -t Positive

