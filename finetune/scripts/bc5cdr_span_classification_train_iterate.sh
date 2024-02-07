#!/bin/bash

set -e
set -u

path_name=$1
iteration=$2
cnt=$3
pre_cnt=$((cnt-1))


HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="bc5cdr"
SUBTASK="span_classification"
MODULE="train_iterate"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

TEMP_CONFIG_DIR="configs/template"
TEMP_YAML_PATH="${TEMP_CONFIG_DIR}/${YAML_FILE}"


#sed -e "s/{path_name}/$path_name/g" $TEMP_YAML_PATH > $YAML_PATH
sed -e "s/{path_name}/$path_name/g;s/{iteration}/$iteration/g;s/{cnt}/$cnt/g;s/{pre_cnt}/$pre_cnt/g" $TEMP_YAML_PATH > $YAML_PATH


if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python3 modules/preprocess/annotation/span_classification_train_iterate.py --yaml $YAML_PATH

