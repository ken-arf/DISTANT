#!/bin/bash

set -e
set -u

timestamp=$1

if [ $timestamp = "" ]; then
    echo "$0 timestamp, timestamp not given"
    exit 1
fi

HOME=$PWD
export PYTHONPATH="$HOME:$HOME/modules"

TASK="immunology"
SUBTASK="span_classification"
MODULE="train"

CONFIG_DIR="configs"
YAML_FILE="${TASK}_${SUBTASK}_${MODULE}.yaml"
YAML_PATH="${CONFIG_DIR}/${YAML_FILE}"
LOG_DIR="experiments/$TASK/logs"
LOG_PATH="${LOG_DIR}/${TASK}_${SUBTASK}_${MODULE}.log"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python3 modules/preprocess/annotation/span_classification_train.py --yaml $YAML_PATH  --timestamp $timestamp


