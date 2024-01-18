#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${__dir}/srun_setup.sh

srun --export=all  -c 1 "$@"