#!/bin/bash

set -e
set -u

path_name=$1
iteration=$2
cnt=$3


HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="bc5cdr"
SUBTASK="misc"
MODULE="convert_brat2conll_iterate"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"


TEMP_CONFIG_DIR="configs/template"
TEMP_YAML_PATH="${TEMP_CONFIG_DIR}/${YAML_FILE}"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

sed -e "s/{path_name}/$path_name/g;s/{iteration}/$iteration/g;s/{cnt}/$cnt/g" $TEMP_YAML_PATH > $YAML_PATH

#nohup python modules/preprocess/abstract/extract_pubmed_abstract.py --yaml $YAML_PATH > $LOG_PATH &
python modules/misc/convert_brat_to_conll.py --yaml $YAML_PATH

#sleep 2
#tail -f $LOG_PATH
