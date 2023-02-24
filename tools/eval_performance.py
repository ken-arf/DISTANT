#!/usr/bin/env python
# coding: utf-8

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

import os
import sys

import pdb

def extract_labels(coll_file):

    with open(coll_file) as fp:
        text = fp.read()

    pdb.set_trace()
    labels = []
    label = []
    sents = text.split('/n')
    for sent in sents:
        if len(sent) == 0:
            labels.append(label)
            label = []
        else:
            label.append(sent.split('\t')[-1])

    if len(label) > 0:
        labels.append(label)

    return labels


y_true = extract_labels('true.coll')
y_pred = extract_labels('predict.coll')


print(f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

