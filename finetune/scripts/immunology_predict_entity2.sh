#!/bin/bash

set -e
set -u

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="immunology"
SUBTASK="predict"
MODULE="entity2"

CONFIG_DIR="configs/finetune"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python modules/finetune/annotation/predict2.py --yaml $YAML_PATH 


