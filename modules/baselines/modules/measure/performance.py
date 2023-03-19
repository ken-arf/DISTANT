import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score

import pdb

def performance_acc(predictions, labels, logger):

    y_pred = predictions
    y_true = labels

    assert len(y_pred) == len(y_true)
    logger.debug(f"event y_true: {y_true}")
    logger.debug(f"event y_pred: {y_pred}")

    if len(y_pred) == 0 and len(y_true) == 0:
        return 1.0

    if type(y_pred[0]) == list and type(y_true[0]) == list:
        acc = accuracy_score(sum(y_true,[]), sum(y_pred,[]))
    else:
        acc = accuracy_score(y_true, y_pred)
    logger.info(f"event acc: {acc}")
    return acc

