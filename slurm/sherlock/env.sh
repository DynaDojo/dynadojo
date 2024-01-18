#!/bin/bash

# SHERLOCK CLUSTER SETTINGS
export DD_SINGULARITY_IMAGE_LOCATION=$GROUP_HOME/$USER/simg     #where you want to store the singularity image
export DD_REPO_DIR=$HOME                                        #parent directory of where you cloned dynadojo
export DD_SCRATCH_DIR=$SCRATCH                                  #your scratch directory  
export DD_OUTPUT_DIR=sherput                                    #name of folder in scratch to put output
export DD_IMAGE_REPO=docker://carynbear/dynadojo:sherlock       #docker image to pull
export DD_CLUSTER=sherlock                                      #cluster name

echo "Setting up environment for $DD_CLUSTER"