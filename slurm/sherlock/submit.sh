#!/bin/bash

# SHERLOCK CLUSTER SETTINGS
PARTITION='-p normal'
OUTPUT='-o ${SCRATCH}/logs/out/%A_%a.out'
ERROR='-e ${SCRATCH}/logs/err/%A_%a.err'
sbatch $PARTITION $OUTPUT $ERROR --export=all "$@"




