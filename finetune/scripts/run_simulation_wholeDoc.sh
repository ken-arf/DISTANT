#!/bin/bash


seed=("1")
ratio=("0.01" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5")

for random_seed in ${seed[@]}; do
    for sample_ratio in ${ratio[@]}; do
        echo "sh ./scripts/run_finetune.sh $sample_ratio $random_seed"
        sh ./scripts/run_finetune_wholeDoc.sh $sample_ratio $random_seed
    done
done

