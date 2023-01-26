#!/bin/bash

set -e
set -u

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="immunology"
SUBTASK="preprocess"
MODULE="dictionary"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

#nohop python modules/preprocess/dictionary/extract_mesh_dictionary.py --yaml $YAML_PATH  > $LOG_PATH &
python modules/preprocess/dictionary/extract_mesh_dictionary.py --yaml $YAML_PATH 

sleep 5
tail -f $LOG_PATH

# postprocess
# delete duplicated items

files=`ls /Users/kenyano/WORK/AIST/Immunology/data/Mesh/dict/*`

for file in $files; do
    cp $file $file.tmp
    sort $file.tmp | uniq > $file
    echo $file
    rm $file.tmp
done
