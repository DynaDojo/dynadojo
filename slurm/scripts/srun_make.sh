
#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color


g() {
  echo -e -n '\033[0;32m'
};
n() {
  echo -e -n '\033[0m'  # No Color
}

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${__dir}/_setup.sh


#ask for challenge, system, and algorithm
read -p "Challenge "fc","fts","fe"]: " challenge
read -p "System: " system
read -p "Algorithm: " algo

#ask for confirmation
g; echo "Run $challenge $system $algo?"; n;
select yn in "Y" "N"; do
    case $yn in
      Y ) break;;
      N )  
           read -p "Challenge "fc","fts","fe"]: " challenge
           read -p "System: " system
           read -p "Algorithm: " algo
           echo "Make $challenge $system $algo?";;
      *) echo "Invalid option. Try again.";continue;;
    esac
done

chmod +x slurm/jobscripts/make.sh #make the script executable
srun $DD_SLURM_ARGS --export=all -c 1 slurm/jobscripts/make.sh $challenge $system $algo