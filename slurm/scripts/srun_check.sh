
#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${__dir}/srun_setup.sh

# Warn user that the directory must be relative to the singularity container
echo "Warning: The data directory must be relative to the singularity container."

#ask for challenge, system, and algorithm
read -p "Directory where data lives? " data_dir

chmod +x slurm/jobscripts/check.sh #make the script executable
srun --export=all -c 1 slurm/jobscripts/check.sh $data_dir