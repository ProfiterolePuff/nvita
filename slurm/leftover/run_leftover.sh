#!/bin/bash

###
# This script submits all leftover experiements at once.
#
# Author: Luke Chang (xcha011@aucklanduni.ac.nz)
# Date:   18/07/2022
###

for I in {1..80}
sbatch ./slurm/leftover/run_$I.sh

squeue --me
