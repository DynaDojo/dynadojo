#!/bin/bash
### CHANGE THESE VALUES TO YOUR OWN
SINGULARITY_IMAGE_LOCATION=$GROUP_HOME/$USER/simg    #where you want to store the singularity image
REPO_DIR=$HOME                            #parent directory of where you cloned dynadojo
SCRATCH_DIR=$SCRATCH                         #your scratch directory  
OUTPUT_DIR=sherput                       #name of folder in scratch to put output
IMAGE_REPO=docker://carynbear/dynadojo:sherlock       #docker image to pull
### CHANGE THESE VALUES TO YOUR OWN

DATA_DIR=$1

singularity run --bind $REPO_DIR/dynadojo/experiments:/dynadojo/experiments \
                --bind $SCRATCH_DIR/$OUTPUT_DIR:/dynadojo/experiments/outputs \
                --bind $REPO_DIR/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif \
                python -u -m experiments \
                    check \
                        --data_dir=$DATA_DIR

# To run: srun --export=all  -c 1 dynadojo/experiments/sherlock/jobscripts/make.sh fc lds lr