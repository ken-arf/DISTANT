#!/bin/bash

ROOT_DIR=../data/BC5CDR/eval
ROOT_DIR=../modules/baselines/data/BC5CDR/eval


predict_coll_dir=$ROOT_DIR/annotate.debug
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

rm tmp
rm *.coll
