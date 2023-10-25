#!/bin/bash
#SBATCH -J lr_lds # A single job name for the array
#SBATCH -n 1 # One Core
#SBATCH -N 1 # All cores on one machine
#SBATCH -p owners # Partition name
#SBATCH --mem 2000 # Memory (2Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o lr_lds_%A_%a.out # Standard output
#SBATCH -e lr_lds_%A_%a.err # Standard error
#SBATCH --array=1-10   # maps 1 to 100 to SLURM_ARRAY_TASK_ID below
