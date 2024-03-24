#!/bin/bash

set -e
set -u

sample_ratio=$1
random_seed=$2
label_weight=$3


#rm -f ./data/bc5cdr/finetune/conll/*
#rm -f ./data/bc5cdr/finetune/span/*

check_status () {
    status=$1
    if [ $status -ne 0 ];then
        echo "failed, exit"
        exit 1
    fi
}

#path_name="Scratch_W${label_weight}_S${sample_ratio}_R${random_seed}"
path_name="W${label_weight}_S${sample_ratio}_R${random_seed}"

#rm -rf ./data/BC5CDR/finetune/$path_name

echo $path_name

sh ./scripts/bc5cdr_prepare_finetuneData_forSimulation.sh $sample_ratio $random_seed $path_name $label_weight chunkUp
check_status $?

#sh ./scripts/bc5cdr_segmentation_train_finetune.sh $path_name  
#check_status $?


#sh ./scripts/bc5cdr_span_classification_train.sh $path_name
#check_status $?


#sh ./scripts/bc5cdr_evaluate_test.sh $path_name
#check_status $?

#sh ./scripts/bc5cdr_misc_convert_brat2conll.sh $path_name
#check_status $?
