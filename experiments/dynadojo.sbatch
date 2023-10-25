#!/bin/bash
#SBATCH -J dynadojo # A single job name for the array
#SBATCH -n 1 # One Core
#SBATCH -N 1 # All cores on one machine
#SBATCH -p normal # Partition name
#SBATCH --mem 2000 # Memory (2Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o outputs/sbatch_log/dynadojo_%A_%a.out # Standard output
#SBATCH -e outputs/sbatch_log/dynadojo_%A_%a.err # Standard error
#SBATCH --array=1-10   # maps 1 to 10 to SLURM_ARRAY_TASK_ID below

CHALLENGE=$1
singularity run --bind $HOME/dynadojo/experiments:/dynadojo/experiments \
                --bind $HOME/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $GROUP_HOME/$USER/simg/dynadojo \
                python -m experiments \
                    --output_dir=$SCRATCH/ \
                    --challenge=$CHALLENGE \
                    --node=$SLURM_ARRAY_TASK_ID \
                    --total_nodes=$SLURM_ARRAY_TASK_MAX \