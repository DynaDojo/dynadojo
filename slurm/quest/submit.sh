#!/bin/bash

# QUEST CLUSTER SETTINGS
ACCOUNT='-A p32141'
PARTITION='-p short'
OUTPUT='-o /home/ctb3982/logs/out/%A_%a.out'
ERROR='/home/ctb3982/logs/err/%A_%a.err'
sbatch $ACCOUNT $PARTITION $OUTPUT $ERROR --export=all "$@"




