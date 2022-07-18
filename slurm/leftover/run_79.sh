#!/bin/bash
#SBATSH --job-name=nvita_leftover
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6

python experiments/step4_attack_non_target.py -d NZTemp -m GRU -a FULLVITA -s 9999 -e 0.15 -n 1
