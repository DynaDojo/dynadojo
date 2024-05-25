#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${__dir}/_setup_caryn.sh

echo "Running interactive singularity shell."
chmod +x slurm/jobscripts/interactive.sh
unset PROMPT_COMMAND
srun $DD_SLURM_ARGS --export=all  -c 1 --pty slurm/jobscripts/interactive.sh