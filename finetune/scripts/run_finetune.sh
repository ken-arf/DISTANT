#!/bin/bash

CONFIG_TRIAL_DIR=./configs/trial

rm -f ./data/bc5cdr/annotate/*
rm -f ./data/bc5cdr/conll/*
rm -f ./data/bc5cdr/database/*
rm -f ./data/bc5cdr/extract/*
rm -f ./data/bc5cdr/finetune/*
rm -f ./data/bc5cdr/span/*
rm -rf ./models/bc5cdr/20*

check_status () {
    status=$1
    if [ $status -ne 0 ];then
        echo "failed, exit"
        exit 1
    fi
}

pubmed_updated_documents=./data/Mesh/PubMed/bc5cdr/finetune/updated_doc.json
pubmed_unchanged_documents=./data/Mesh/PubMed/bc5cdr/finetune/unchanged_doc.json

path=`dirname $pubmed_updated_documents`
if [ ! -d $path ]; then
    mkdir -p $path
fi

#sh ./scripts/search_compare_datefield.sh > $pubmed_updated_documents 
echo "extracting updated documents"
sh ./scripts/search_documents.sh updateOnly > $pubmed_updated_documents
check_status $?

echo "extracting unchaged documents"
sh ./scripts/search_documents.sh all > $pubmed_unchanged_documents
check_status $?

sh ./scripts/immunology_prepare_finetuneData.sh
check_status $?


timestamp=`date '+%Y%m%d_%H%M%S'` 

sh ./scripts/immunology_segmentation_train_finetune.sh $timestamp
check_status $?

sh ./scripts/immunology_span_classification_train.sh $timestamp
check_status $?

# output config file for annotation for trial
cat $CONFIG_TRIAL_DIR/immunology_predict_entity2_template.yaml | sed -e "s/{timestamp}/$timestamp/" > $CONFIG_TRIAL_DIR/immunology_predict_entity2.yaml

cat $CONFIG_TRIAL_DIR/immunology_segmentation_predict_template.yaml | sed -e "s/{timestamp}/$timestamp/" > $CONFIG_TRIAL_DIR/immunology_segmentation_predict.yaml

