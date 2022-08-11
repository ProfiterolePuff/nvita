#!/bin/bash

for I in {1..544}; do
    sbatch ./new_slurm/part_$I.sh
done
        
squeue --me
