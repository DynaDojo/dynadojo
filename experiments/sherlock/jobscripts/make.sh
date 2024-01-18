#!/bin/bash
#SBATCH -J dynadojo_make
#SBATCH -n 2 # Cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -p normal # Partition name
#SBATCH --mem 500 # Memory (500mb)
#SBATCH -t 0-0:15 # Maximum execution time (D-HH:MM)
#SBATCH -o logs/out/%A_%a.out # Standard output
#SBATCH -e logs/out/%A_%a.out # Standard error

### CHANGE THESE VALUES TO YOUR OWN
SINGULARITY_IMAGE_LOCATION=$GROUP_HOME/$USER/simg    #where you want to store the singularity image
REPO_DIR=$HOME                            #parent directory of where you cloned dynadojo
SCRATCH_DIR=$SCRATCH                         #your scratch directory  
OUTPUT_DIR=sherput                       #name of folder in scratch to put output
### CHANGE THESE VALUES TO YOUR OWN

CHALLENGE=$1
SYSTEM=$2
ALGO=$3

singularity run --bind $REPO_DIR/dynadojo/experiments:/dynadojo/experiments \
                --bind $SCRATCH_DIR/$OUTPUT_DIR:/dynadojo/experiments/outputs \
                --bind $REPO_DIR/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif \
                python -u -m experiments \
                    make \
                        --challenge=$CHALLENGE \
                        --system=$SYSTEM \
                        --algo=$ALGO

# To run: srun --export=all  -c 1 dynadojo/experiments/sherlock/jobscripts/make.sh fc lds lr