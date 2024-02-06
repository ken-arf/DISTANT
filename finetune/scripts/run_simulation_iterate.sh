#!/bin/bash


seed=("1")
declare -a ratio_iteration=("0.1 1" "0.2 2" "0.3 3" "0.4 4" "0.5 5")
#ratio=("0.1")
#label_weights=("2" "3" "4" "5" "6" "7" "8" "9" "10")
label_weight=6

for random_seed in "${seed[@]}"; do
    for sample_ratio in "${ratio_iteration[@]}"; do
        set -- $sample_ratio
        ratio=$1
        iteration=$2
        loop=`seq 1 $iteration`
        for cnt in $loop; do
            echo "sh ./scripts/run_finetune_iterate.sh $ratio $random_seed $label_weight $iteration $cnt"
        done
    done
done

