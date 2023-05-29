#!/bin/bash

set -e
set -u

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="ncbi"
SUBTASK="misc"
MODULE="convert_txt"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

#nohup python modules/preprocess/abstract/extract_pubmed_abstract.py --yaml $YAML_PATH > $LOG_PATH &
python modules/misc/convert_NCBI.py --yaml $YAML_PATH

sleep 2
tail -f $LOG_PATH
