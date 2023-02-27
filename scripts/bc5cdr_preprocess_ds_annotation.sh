#!/bin/bash

set -e
set -u

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="bc5cdr"
SUBTASK="preprocess"
MODULE="ds_annotation"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

DRIVER_MEMORY=2G
#nohup python modules/preprocess/annotation/ds_annotation.py --yaml $YAML_PATH > $LOG_PATH &
#python modules/preprocess/annotation/ds_annotation.py --yaml $YAML_PATH 
spark-submit --driver-memory $DRIVER_MEMORY modules/preprocess/annotation/ds_annotation_spark.py --yaml $YAML_PATH 

#sleep 5
#tail -f $LOG_PATH

