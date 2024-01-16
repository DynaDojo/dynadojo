#!/bin/bash

# First, request an interactive session, run:
# bash /experiments/quest/scripts/session.sh

### CHANGE THESE VALUES TO YOUR OWN
SINGULARITY_IMAGE_LOCATION=$GROUP_HOME/$USER/simg     #where you want to store the singularity image
REPO_DIR=$HOME                                        #parent directory of where you cloned dynadojo
SCRATCH_DIR=$SCRATCH                                  #your scratch directory  
OUTPUT_DIR=sherput                                   #name of folder in scratch to put output
### CHANGE THESE VALUES TO YOUR OWN

# Interactively run singularity
module load singularity

if test -f $SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif; then
  echo “docker image exists”
else
    mkdir -p $SINGULARITY_IMAGE_LOCATION
    singularity pull $SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif docker://carynbear/dynadojo:sherlock
fi

singularity shell \
                --bind $SCRATCH_DIR/$OUTPUT_DIR:/$OUTPUT_DIR \
                --bind $REPO_DIR/dynadojo/experiments:/dynadojo/experiments \
                --bind $REPO_DIR/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif

# Now you can use the Experiments command line interface to make and check experiments