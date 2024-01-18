#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash ${__dir}/srun_setup.sh

echo "Running interactive singularity shell."
chmod +x slurm/jobscripts/interactive.sh
unset PROMPT_COMMAND
srun --export=all  -c 1 --pty slurm/jobscripts/interactive.sh