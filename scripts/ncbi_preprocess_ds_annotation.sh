#!/bin/bash

set -e
set -u

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="ncbi"
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

spark-submit --driver-memory $DRIVER_MEMORY modules/preprocess/annotation/ds_annotation_spark_test.py --yaml $YAML_PATH 

#python modules/preprocess/annotation/ds_annotation_normal.py --yaml $YAML_PATH

#sleep 5
#tail -f $LOG_PATH

