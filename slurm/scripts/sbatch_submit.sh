#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${__dir}/_setup.sh


sbatch $DD_SLURM_ARGS $DD_SLURM_SAVE --export=all "$@"