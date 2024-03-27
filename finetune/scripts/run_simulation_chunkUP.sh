#!/bin/bash


seed=("1")
#ratio=("0.01" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5")
#ratio=("0.1" "0.2" "0.3" "0.4" "0.5")
ratio=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")


ratio=("0.8" "0.9" "1.0")
#ratio=("0.1")
#label_weights=("2" "3" "4" "5" "6" "7" "8" "9" "10")
label_weight=1

for random_seed in ${seed[@]}; do
    for sample_ratio in ${ratio[@]}; do
        #for label_weight in ${label_weights[@]}; do
            #echo "sh ./scripts/run_finetune.sh $sample_ratio $random_seed $label_weight"
            sh ./scripts/run_finetune_chunkUp.sh $sample_ratio $random_seed $label_weight
            #sh ./scripts/run_finetune_scratch.sh $sample_ratio $random_seed $label_weight
        #done
    done
done

