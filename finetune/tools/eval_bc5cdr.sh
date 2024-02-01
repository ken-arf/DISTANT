#!/bin/bash

ROOT_DIR=../data/BC5CDR/eval


annotate_paths=$(ls ../data/BC5CDR/eval | grep annotate | grep -v gold)

predict_dir=()

for path in $annotate_paths; do
    predict_dir+=(""$path"")
done

echo ${predict_dir[*]}

#predict_dir=("annotate" "annotate.20240130_215056" "annotate.240128" "annotate.240128_2" "annotate.1" "annotate.2" "annotate.3" "annotate.autoner" "annotate.match_dict" "annotate.supervised")
#predict_dir+=("annotate.bond2nd" "annotate.roster")

for i in ${!predict_dir[@]}; do
    echo "$i: ${predict_dir[$i]}"
done

echo "select>"
read input

hypo=${predict_dir[$input]}
predict_coll_dir=$ROOT_DIR/${hypo}
echo "hyp dir: ${predict_coll_dir}"

#predict_coll_di=$ROOT_DIR/annotate.latest
#predict_coll_dir=$ROOT_DIR/annotate.match_dict
#predict_coll_dir=$ROOT_DIR/annotate
#predict_coll_dir=$ROOT_DIR/annotate.autoner
#predict_coll_dir=$ROOT_DIR/annotate.supervised

true_coll_dir=$ROOT_DIR/annotate.gold
echo "ref dir: ${true_coll_dir}"


ls $predict_coll_dir/*.coll | sort | xargs -n1 cat > predict.coll
ls $true_coll_dir/*.coll | sort | xargs -n1 cat > true.coll


python eval_performance.py

rm *.coll
