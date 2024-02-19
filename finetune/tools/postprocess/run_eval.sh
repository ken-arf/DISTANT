#!/bin/bash

input="folder_names.txt"

result_dir=./results

rm -fr $result_dir

mkdir $result_dir

while IFS= read -r line
do
    if [[ $line =~ "annotate" ]]; then 
        echo "$line"
        sh eval_f1.sh $line > $result_dir/$line.f1
    fi
done < "$input"

