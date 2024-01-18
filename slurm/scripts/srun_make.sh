
#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash ${__dir}/srun_setup.sh


#ask for challenge, system, and algorithm
read -p "Challenge "fc","fts","fe"]: " challenge
read -p "System: " system
read -p "Algorithm: " algo

#ask for confirmation
g; echo "Run $challenge $system $algo?"; n;
select yn in "yes" "no"; do
    case $yn in
      Yes ) break;;
      No ) 
           read -p "Challenge "fc","fts","fe"]: " challenge
           read -p "System: " system
           read -p "Algorithm: " algo
           echo "Run $challenge $system $algo?";;
      *) echo "Invalid option. Try again.";continue;;
    esac
done

chmod +x slurm/jobscripts/make.sh #make the script executable
srun --export=all -c 1 slurm/jobscripts/make.sh $challenge $system $algo