#!/bin/bash

###
# This script submits all experiements at once.
#
# Author: Luke Chang (xcha011@aucklanduni.ac.nz)
# Date:   14/07/2022
###
sbatch ./slurm/run_cny.sh
sbatch ./slurm/run_cny_target.sh
sbatch ./slurm/run_electricity.sh
sbatch ./slurm/run_electricity_target.sh
sbatch ./slurm/run_nztemp.sh
sbatch ./slurm/run_nztemp_target.sh
sbatch ./slurm/run_oil.sh
sbatch ./slurm/run_oil_target.sh

squeue --me
