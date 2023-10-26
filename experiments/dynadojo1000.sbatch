#!/bin/bash

#SBATCH -n 1 # Num Cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -p normal # Partition name
#SBATCH --mem-per-cpu=8G # Memory (4Gb)
#SBATCH -t 0-3:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs/out/%A_%a.out # Standard output
#SBATCH -e logs/err/%A_%a.err # Standard error
#SBATCH --array=1-500   # -10 maps 1 to 10 to SLURM_ARRAY_TASK_ID below
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=mkanwal@stanford.edu

CHALLENGE=$1
SYSTEM=$2
MODEL=$3

singularity run --bind $HOME/dynadojo/experiments:/dynadojo/experiments \
                --bind $HOME/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $GROUP_HOME/$USER/simg/dynadojo \
                python -m experiments \
                    --output_dir=$SCRATCH/ \
                    --system=$SYSTEM \
                    --model=$MODEL \
                    --challenge=$CHALLENGE \
                    --node=$SLURM_ARRAY_TASK_ID \
                    --total_nodes=$SLURM_ARRAY_TASK_MAX \

# To run: sbatch -J dynadojo dynadojo1000.sbatch fe lds dnn_simple
# To get report: smanage report --sacct --name=dynadojo
# To get details: squeue --user=mkanwal
# To cancel: scancel [jobid]
