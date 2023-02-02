#!/bin/bash

set -e
set -u

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="immunology"
SUBTASK="preprocess"
MODULE="convert_annotation"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

#nohup python modules/preprocess/annotation/convert_csv_to_brat.py --yaml $YAML_PATH > $LOG_PATH &
python modules/preprocess/annotation/convert_csv_to_brat.py --yaml $YAML_PATH 

sleep 5
tail -f $LOG_PATH

