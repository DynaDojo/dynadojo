#!/bin/bash
#SBATCH -A p32141
#SBATCH -J dynadojo_make
#SBATCH -n 2 # Four Cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -p short # Partition name
#SBATCH --mem 500 # Memory (500mb)
#SBATCH -t 0-0:10 # Maximum execution time (D-HH:MM)
#SBATCH -o /home/ctb3982/logs/out/%A_%a.out # Standard output
#SBATCH -e /home/ctb3982/logs/err/%A_%a.err # Standard error

### CHANGE THESE VALUES TO YOUR OWN
SINGULARITY_IMAGE_LOCATION=$HOME/simg     #where you want to store the singularity image
REPO_DIR=$HOME                            #parent directory of where you cloned dynadojo
SCRATCH_DIR=$HOME                         #your scratch directory  
OUTPUT_DIR=questput                       #name of folder in scratch to put output
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

# To run: sbatch make.sbatch fc lds lr