#!/bin/bash

set -e
set -u

sample_ratio=$1
random_seed=$2


#rm -f ./data/bc5cdr/finetune/conll/*
#rm -f ./data/bc5cdr/finetune/span/*

check_status () {
    status=$1
    if [ $status -ne 0 ];then
        echo "failed, exit"
        exit 1
    fi
}

path_name="WDoc_noPretrain_S${sample_ratio}_R${random_seed}"

rm -rf ./data/BC5CDR/finetune/$path_name


sh ./scripts/bc5cdr_prepare_finetuneData_forSimulation_wholeDocument.sh $sample_ratio $random_seed $path_name
check_status $?

sh ./scripts/bc5cdr_segmentation_train_finetune_noPretrain.sh $path_name
check_status $?

sh ./scripts/bc5cdr_span_classification_train_noPretrain.sh $path_name
check_status $?

sh ./scripts/bc5cdr_evaluate_test.sh $path_name
check_status $?

sh ./scripts/bc5cdr_misc_convert_brat2conll.sh $path_name
check_status $?
