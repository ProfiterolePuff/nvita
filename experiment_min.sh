#!/bin/bash
################################################################################
# This script runs the minimal demo experiment
#
# Author: Luke Chang (xcha011@aucklanduni.ac.nz)
# Date:   14/07/2022
################################################################################


SEED="58361"
DATA="Oil"
MODEL="CNN"
ATTACKS=("NOATTACK" "FGSM" "FULLVITA")
EPS=0.2

for ATTACK in ${ATTACKS[@]}; do
    python experiments/step4_attack_non_target.py -d $DATA  -m $MODEL -a $ATTACK -s $SEED -e $EPS -n 1 --demo 10
done
