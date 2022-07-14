#!/bin/bash
################################################################################
# This script is for running the full experiments.
# Each seed value indicates one experiment run.
# "--demo" is an optional parameter to run the experiment with only 10 samples. 
# E.g., add "--demo 10" 
#
# Author: Luke Chang (xcha011@aucklanduni.ac.nz)
# Date:   14/07/2022
################################################################################


# SEEDS=("2210" "9999" "58361" "789789" "1111111")
# DATASETS=("Electricity" "NZTemp" "CNYExch" "Oil")
SEEDS=("9999")
DATASETS=("CNYExch")
MODELS=("CNN" "LSTM" "GRU" "RF")
ATTACKS_UNTARGETED=("NOATTACK" "BRS" "FGSM" "BIM" "FULLVITA")
# PARAM_N=(1 3 5) # Only used in NVITA
PARAM_N=(1 5) # Only used in NVITA
# EPSILONS=(0.05 0.1 0.15 0.20)
EPSILONS=(0.20)
TARGETS=("Positive" "Negative")


# For non-targeted attacks
for SEED in ${SEEDS[@]}; do
    for EPS in ${EPSILONS[@]}; do
        for DATA in ${DATASETS[@]}; do
            for MODEL in ${MODELS[@]}; do
                for ATTACK in ${ATTACKS_UNTARGETED[@]}; do
                    python experiments/step4_attack_non_target.py -d $DATA  -m $MODEL -a $ATTACK -s $SEED -e $EPS -n 1 --demo 10
                done

                # Only for NVITA
                for N in ${PARAM_N[@]}; do
                    python experiments/step4_attack_non_target.py -d $DATA  -m $MODEL -a NVITA -s $SEED -e $EPS -n $N --demo 10
                done

                # n=5 for BRNV
                python experiments/step4_attack_non_target.py -d $DATA  -m $MODEL -a BRNV -s $SEED -e $EPS -n 5 --demo 10
            done
        done
    done
done


# For targeted attacks
ATTACKS_TARGETED=("NOATTACK" "FULLVITA")

for SEED in ${SEEDS[@]}; do
    for EPS in ${EPSILONS[@]}; do
        for DATA in ${DATASETS[@]}; do
            for MODEL in ${MODELS[@]}; do
                for TARGET in ${TARGETS[@]}; do
                    for ATTACK in ${ATTACKS_TARGETED[@]}; do
                        python experiments/step5_attack_target.py -d $DATA  -m $MODEL -a $ATTACK -t $TARGET -s $SEED -e $EPS -n 1 --demo 10
                    done

                    # Only for NVITA
                    for N in ${PARAM_N[@]}; do
                        python experiments/step5_attack_target.py -d $DATA  -m $MODEL -a NVITA -t $TARGET -s $SEED -e $EPS -n $N --demo 10
                    done

                    # n=5 for BRNV
                    python experiments/step5_attack_target.py -d $DATA  -m $MODEL -a BRNV -t $TARGET -s $SEED -e $EPS -n 5 --demo 10
                done
            done
        done
    done
done
