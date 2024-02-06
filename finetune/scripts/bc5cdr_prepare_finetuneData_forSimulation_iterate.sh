#!/bin/bash

set -e
set -u

sample_ratio=$1
random_seed=$2
path_name=$3
label_weight=$4

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="bc5cdr"
SUBTASK="prepare"
MODULE="finetuneData_forSimulation"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

TEMP_CONFIG_DIR="configs/template"
TEMP_YAML_PATH="${TEMP_CONFIG_DIR}/${YAML_FILE}"


#sed -e "s/{sample_ratio}/$sample_ratio/g" $YAML_PATH
#sed -e "s/{random_seed}/$random_seed/g" $YAML_PATH

sed -e "s/{sample_ratio}/$sample_ratio/g;s/{random_seed}/$random_seed/g;s/{path_name}/$path_name/g;s/{label_weight}/$label_weight/g" $TEMP_YAML_PATH > $YAML_PATH

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python3 modules/database/prepare_finetuneData_forSimulation.py --yaml $YAML_PATH 

