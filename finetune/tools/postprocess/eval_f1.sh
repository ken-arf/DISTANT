#!/bin/bash


set -e
set -u

hypo=$1

ROOT_DIR=../../data/BC5CDR/eval

predict_coll_dir=$ROOT_DIR/${hypo}
echo "hyp dir: ${predict_coll_dir}"


true_coll_dir=$ROOT_DIR/annotate.gold
echo "ref dir: ${true_coll_dir}"


ls $predict_coll_dir/*.coll | sort | xargs -n1 cat > predict.coll
ls $true_coll_dir/*.coll | sort | xargs -n1 cat > true.coll


python ../eval_performance.py

rm *.coll
