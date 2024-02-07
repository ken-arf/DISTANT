#!/bin/bash

set -e
set -u

path_name=$1
iteration=$2
cnt=$3

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="bc5cdr"
SUBTASK="evaluate"
MODULE="test_iterate"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"

YAML_FILE2="bc5cdr_segmentation_predict.yaml"
YAML_PATH2="${CONFIG_DIR}/${YAML_FILE2}"

LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi


TEMP_CONFIG_DIR="configs/template"
TEMP_YAML_PATH="${TEMP_CONFIG_DIR}/${YAML_FILE}"
TEMP_YAML_PATH2="${TEMP_CONFIG_DIR}/${YAML_FILE2}"


#sed -e "s/{path_name}/$path_name/g" $TEMP_YAML_PATH > $YAML_PATH
#sed -e "s/{path_name}/$path_name/g" $TEMP_YAML_PATH2 > $YAML_PATH2
sed -e "s/{path_name}/$path_name/g;s/{iteration}/$iteration/g;s/{cnt}/$cnt/g" $TEMP_YAML_PATH > $YAML_PATH
sed -e "s/{path_name}/$path_name/g;s/{iteration}/$iteration/g;s/{cnt}/$cnt/g" $TEMP_YAML_PATH2 > $YAML_PATH2


#nohup python modules/preprocess/annotation/ds_annotation.py --yaml $YAML_PATH > $LOG_PATH &
python modules/preprocess/annotation/pu_predict.py --yaml $YAML_PATH 

#sleep 5
#tail -f $LOG_PATH

