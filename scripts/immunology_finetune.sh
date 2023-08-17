#!/bin/bash

set -e
set -u

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="immunology"
SUBTASK="finetune"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi


python modules/finetune/segmentation/make_dataset2.py --yaml $YAML_PATH 


