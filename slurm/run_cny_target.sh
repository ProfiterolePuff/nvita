#!/bin/bash
#SBATSH --job-name=nvita_CNYExch_targeted
#SBATCH --time=30:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4

################################################################################
# Targeted attacks on CNYExch
#
# Author: Luke Chang (xcha011@aucklanduni.ac.nz)
# Date:   14/07/2022
################################################################################

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03620/nvita/venv/bin/activate

DATA="CNYExch"
SEEDS=("2210" "9999" "58361" "789789" "1111111")
MODELS=("CNN" "LSTM" "GRU" "RF")
ATTACKS=("NOATTACK" "BRS" "FGSM" "BIM" "FULLVITA")
PARAM_N=(1 3 5) # Only used in NVITA
EPSILONS=(0.05 0.1 0.15 0.2)
TARGETS=("Positive" "Negative")

# For targeted attacks
for EPS in ${EPSILONS[@]}; do
    for MODEL in ${MODELS[@]}; do
        for TARGET in ${TARGETS[@]}; do
            for ATTACK in ${ATTACKS[@]}; do
                python experiments/step5_attack_target.py -d $DATA  -m $MODEL -a $ATTACK -t $TARGET -s ${SEEDS[$SLURM_ARRAY_TASK_ID]} -e $EPS -n 1
            done

            # Only for NVITA
            for N in ${PARAM_N[@]}; do
                python experiments/step5_attack_target.py -d $DATA  -m $MODEL -a NVITA -t $TARGET -s ${SEEDS[$SLURM_ARRAY_TASK_ID]} -e $EPS -n $N
            done

            # n=5 for BRNV
            python experiments/step5_attack_target.py -d $DATA  -m $MODEL -a BRNV -t $TARGET -s ${SEEDS[$SLURM_ARRAY_TASK_ID]} -e $EPS -n 5
        done
    done
done
