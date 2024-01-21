#!/bin/bash

REPO_DIR=$(pwd)
if [[ $REPO_DIR != */dynadojo ]]; then
  echo -e "${RED} Please run this script from the 'dynadojo' directory ${NC}";
  return 0 2> /dev/null || exit 0
fi;

if [[ $DD_CLUSTER == "" ]]; then
  #ask which cluster is the user on "sherlock" or "quest":
  read -p "Which cluster are you on [sherlock,quest]: " cluster
  #repeat until user enters a valid cluster
  while [[ $cluster != "sherlock" && $cluster != "quest" ]]; do
    read -p "Which cluster are you on [sherlock,quest]: " cluster
  done

  export DD_CLUSTER=$cluster
fi;

ENV_SCRIPT="slurm/$DD_CLUSTER/env.sh"
#check if the submit script exists
if [ ! -f "$ENV_SCRIPT" ]; then
  echo -e "${RED} Environment configuration script $ENV_SCRIPT does not exist ${NC}";
  return 0 2> /dev/null || exit 0
else
    echo "Running $ENV_SCRIPT"
    chmod +x $ENV_SCRIPT #make the script executable
    source $ENV_SCRIPT
fi