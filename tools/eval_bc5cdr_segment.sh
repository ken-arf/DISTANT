#!/bin/bash

ROOT_DIR=../data/BC5CDR/eval
ROOT_DIR=../modules/baselines/data/BC5CDR/eval


predict_dir=("annotate.debug.1" "annotate.debug.2" "annotate.debug.3")

for i in ${!predict_dir[@]}; do
    echo "$i: ${predict_dir[$i]}"
done

echo "select>"
read input

hypo=${predict_dir[$input]}
predict_coll_dir=$ROOT_DIR/${hypo}
echo "hyp dir: ${predict_coll_dir}"

#predict_coll_dir=$ROOT_DIR/annotate.debug
#echo "hyp dir: ${predict_coll_dir}"


true_coll_dir=$ROOT_DIR/annotate.gold
echo "ref dir: ${true_coll_dir}"


ls $predict_coll_dir/*.coll | sort | xargs -n1 cat > predict.coll
ls $true_coll_dir/*.coll | sort | xargs -n1 cat > true.coll


for f in predict.coll true.coll; do
    cp $f tmp
    for tag in Disease Chemical NONE; do
        echo $tag
        sed -i -e "s/B\-${tag}/I\-B/g" tmp
        sed -i -e "s/I\-${tag}/I\-B/g" tmp
    done
    mv tmp $f
done


python eval_performance.py

rm *.coll
