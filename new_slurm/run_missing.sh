#!/bin/bash

for I in {1..544}; do
    sbatch ./slurm/missing/part_$I.sh
done
        
squeue --me
