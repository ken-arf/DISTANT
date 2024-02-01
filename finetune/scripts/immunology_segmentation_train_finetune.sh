#!/bin/bash

set -e
set -u

path_name=$1

if [ $timestamp = "" ]; then
    echo "$0 timestamp, timestamp not given"
    exit 1
fi


HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="immunology"
SUBTASK="segmentation"
MODULE="train_finetune"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

sed -i -e "s/{path_name}/$path_name/g" $YAML_PATH

echo $YAML_PATH
python3 modules/preprocess/segmentation/train_finetune.py --yaml $YAML_PATH 

