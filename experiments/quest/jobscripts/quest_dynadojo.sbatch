#!/bin/bash
#SBATCH -A p32141
#SBATCH -J dynadojo # A single job name for the array
#SBATCH -n 2 # Four Cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -p short # Partition name
#SBATCH --mem 7G # Memory (4Gb)
#SBATCH -t 0-2:30 # Maximum execution time (D-HH:MM)
#SBATCH -o /home/ctb3982/logs/out/%A_%a.out # Standard output
#SBATCH -e /home/ctb3982/logs/err/%A_%a.err # Standard error
#SBATCH --array=1-500   # -10 maps 1 to 10 to SLURM_ARRAY_TASK_ID below


CHALLENGE=$1
SYSTEM=$2
MODEL=$3
ACCOUNT=p32141

module load singularity

singularity run --bind $HOME/questput:/questput \
                --bind $HOME/dynadojo/experiments:/dynadojo/experiments \
                --bind $HOME/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $HOME/simg/dynadojo_sherlock.sif \
                python -u -m experiments \
                    --output_dir=/questput \
                    --system=$SYSTEM \
                    --model=$MODEL \
                    --challenge=$CHALLENGE \
                    --node=$SLURM_ARRAY_TASK_ID \
                    --total_nodes=$SLURM_ARRAY_TASK_MAX 

# To run: sbatch quest_dynadojo.sbatch fe lds lr
# To get report: smanage report --sacct --name=dynadojo
# To get details: squeue --user=mkanwal
#$SLURM_ARRAY_TASK_ID \
#$SLURM_ARRAY_TASK_MAX \

#Interactive
#srun -c 4 -p normal -A p32141 -t 0-1:00 --pty bash