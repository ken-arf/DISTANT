#!/bin/bash

set -e
set -u

path_name=$1


HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules:$HOME/modules_aux"

TASK="bc5cdr"
SUBTASK="segmentation"
MODULE="train_finetune"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"


TEMP_CONFIG_DIR="configs/template"
TEMP_YAML_PATH="${TEMP_CONFIG_DIR}/${YAML_FILE}"

sed -e "s/{path_name}/$path_name/g" $TEMP_YAML_PATH > $YAML_PATH


if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

echo $YAML_PATH
python3 modules/preprocess/segmentation/train_finetune.py --yaml $YAML_PATH 

