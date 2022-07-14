#!/bin/bash -e

#SBATSH --job-name=nvita_test_01
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

SEED="58361"
DATA="Oil"
MODEL="CNN"
ATTACKS=("NOATTACK" "FGSM" "FULLVITA")
EPS=0.2

for ATTACK in ${ATTACKS[@]}; do
    python experiments/step4_attack_non_target.py -d $DATA  -m $MODEL -a $ATTACK -s $SEED -e $EPS -n 1 --demo 10
done
