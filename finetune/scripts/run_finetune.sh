#!/bin/bash

CONFIG_TRIAL_DIR=./configs/trial

#rm -f ./data/bc5cdr/finetune/annotate/*
#rm -f ./data/bc5cdr/finetune/conll/*
#rm -f ./data/bc5cdr/finetune/span/*

check_status () {
    status=$1
    if [ $status -ne 0 ];then
        echo "failed, exit"
        exit 1
    fi
}




sh ./scripts/bc5cdr_prepare_finetuneData.sh
check_status $?


timestamp=`date '+%Y%m%d_%H%M%S'` 

sh ./scripts/bc5cdr_segmentation_train_finetune.sh $timestamp
check_status $?

sh ./scripts/bc5cdr_span_classification_train.sh $timestamp
check_status $?

# output config file for annotation for trial
#cat $CONFIG_TRIAL_DIR/bc5cdr_predict_entity2_template.yaml | sed -e "s/{timestamp}/$timestamp/" > $CONFIG_TRIAL_DIR/bc5cdr_predict_entity2.yaml

#cat $CONFIG_TRIAL_DIR/bc5cdr_segmentation_predict_template.yaml | sed -e "s/{timestamp}/$timestamp/" > $CONFIG_TRIAL_DIR/bc5cdr_segmentation_predict.yaml

