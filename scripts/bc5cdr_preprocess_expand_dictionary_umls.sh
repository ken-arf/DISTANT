#!/bin/bash

set -e
set -u

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="bc5cdr"
SUBTASK="preprocess"
MODULE="expand_dictionary_umls"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

#nohup python modules/preprocess/dictionary/expand_umls_dictionary.py --yaml $YAML_PATH > $LOG_PATH &
python modules/preprocess/dictionary/expand_umls_dictionary.py --yaml $YAML_PATH 


