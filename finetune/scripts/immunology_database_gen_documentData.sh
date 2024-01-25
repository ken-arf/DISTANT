#!/bin/bash

set -e
set -u

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="immunology"
SUBTASK="database"
MODULE="gen_documentData"

CONFIG_DIR="configs/finetune"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python modules/database/gen_documentData.py --yaml $YAML_PATH 

#sleep 5
#tail -f $LOG_PATH

