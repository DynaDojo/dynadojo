#!/bin/bash

# SHERLOCK CLUSTER SETTINGS
PARTITION='-p normal'
OUTPUT='-o /logs/out/%A_%a.out'
ERROR='/logs/err/%A_%a.err'
sbatch $PARTITION $OUTPUT $ERROR --export=all "$@"




