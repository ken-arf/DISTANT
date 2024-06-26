#!/bin/bash

set -e
set -u

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="ncbi"
SUBTASK="bond"
MODULE="makedataset"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

#python modules/autoner/make_dataset.py --yaml $YAML_PATH 
python modules/bond/make_dataset.py --yaml $YAML_PATH 

#sleep 5
#tail -f $LOG_PATH

# postprocess
# delete duplicated items

