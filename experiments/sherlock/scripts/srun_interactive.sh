REPO_DIR=$(pwd)
if [[ $REPO_DIR != */dynadojo ]]; then
  echo -e "${RED} Please run this script from the 'dynadojo' directory ${NC}";
  return 0 2> /dev/null || exit 0
fi;

chmod +x experiments/sherlock/jobscripts/interactive.sh
unset PROMPT_COMMAND
srun --export=all  -c 1 --pty experiments/sherlock/jobscripts/interactive.sh