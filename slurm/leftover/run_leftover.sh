#!/bin/bash

###
# This script submits all experiements with Epsilon=0.15 and 0.2.
#
# Author: Luke Chang (xcha011@aucklanduni.ac.nz)
# Date:   15/07/2022
###

sbatch ./slurm/leftover/run_cny_leftover.sh
sbatch ./slurm/leftover/run_cny_target_leftover.sh
sbatch ./slurm/leftover/run_electricity_leftover.sh
sbatch ./slurm/leftover/run_electricity_target_leftover.sh
sbatch ./slurm/leftover/run_nztemp_leftover.sh
sbatch ./slurm/leftover/run_nztemp_target_leftover.sh
sbatch ./slurm/leftover/run_oil_leftover.sh
sbatch ./slurm/leftover/run_oil_target_leftover.sh

squeue --me
