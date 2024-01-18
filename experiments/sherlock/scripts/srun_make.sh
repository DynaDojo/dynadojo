REPO_DIR=$(pwd)
if [[ $REPO_DIR != */dynadojo ]]; then
  echo -e "${RED} Please run this script from the 'dynadojo' directory ${NC}";
  return 0 2> /dev/null || exit 0
fi;

read -p "Challenge "fc","fts","fe"]: " challenge
read -p "System: " system
read -p "Algorithm: " algo

chmod +x experiments/sherlock/jobscripts/make.sh #make the script executable
srun --export=all -c 1 experiments/sherlock/jobscripts/make.sh $challenge $system $algo