#!/bin/bash


ROOT_DIR=../data/BC5CDR/eval

#predict_coll_dir=$ROOT_DIR/annotate.prev
predict_coll_dir=$ROOT_DIR/annotate.latest
predict_coll_dir=$ROOT_DIR/annotate
true_coll_dir=$ROOT_DIR/annotate.gold

ls $predict_coll_dir/*.coll | sort | xargs -n1 cat > predict.coll
ls $true_coll_dir/*.coll | sort | xargs -n1 cat > true.coll


sed -e "s/^ /_/g" -i predict.coll
sed -e "s/^ /_/g" -i true.coll

sed -e "s/B_/B-/g" -i predict.coll
sed -e "s/I_/I-/g" -i predict.coll

sed -e "s/B_/B-/g" -i true.coll
sed -e "s/I_/I-/g" -i true.coll


python eval_performance.py
