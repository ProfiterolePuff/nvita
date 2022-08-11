#!/bin/bash

for I in {1..544}; do
    sbatch ./slurm/negtar/run_$I.sh
done
        
squeue --me
