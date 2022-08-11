#!/bin/bash
#SBATSH --job-name=nvita_Oil_targeted
#SBATCH --output=log/log_%x_%j_%a.out
#SBATCH --error=log/log_%x_%j_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-4

################################################################################
# Negative Targeted attacks on Oil
#
# Author: Luke Chang (xcha011@aucklanduni.ac.nz)
# Date:   11/08/2022
################################################################################

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

DATA="Oil"
TARGET="Negative"
SEEDS=("2210" "9999" "58361" "789789" "1111111")
MODELS=("CNN" "LSTM" "GRU" "RF")
ATTACKS=("NOATTACK" "BRS" "FGSM" "BIM" "FULLVITA")
PARAM_N=(1 3 5) # Only used in NVITA
EPSILONS=(0.05 0.1 0.15 0.2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Seed value = $SEED"

# For negative targeted attacks
for EPS in ${EPSILONS[@]}; do
    for MODEL in ${MODELS[@]}; do
        for ATTACK in ${ATTACKS[@]}; do
            python experiments/step5_attack_target.py -d $DATA  -m $MODEL -a $ATTACK -t $TARGET -s $SEED -e $EPS -n 1
        done

        # Only for NVITA
        for N in ${PARAM_N[@]}; do
            python experiments/step5_attack_target.py -d $DATA  -m $MODEL -a NVITA -t $TARGET -s $SEED -e $EPS -n $N
        done

        # n=5 for BRNV
        python experiments/step5_attack_target.py -d $DATA  -m $MODEL -a BRNV -t $TARGET -s $SEED -e $EPS -n 5
    done
done
