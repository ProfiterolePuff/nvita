#!/bin/bash
#SBATSH --job-name=nvita_leftover
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

python experiments/step5_attack_target.py -d Oil -m CNN -a NOATTACK -s 1111111 -e 0.05 -n 1 -t Positive
