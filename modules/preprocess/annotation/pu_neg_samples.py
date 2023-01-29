

import os
import pickle
import numpy as np
import pandas as pd



def percentile(prob, percent):
    t = np.percentile(prob, percent)
    #print(f"percentile ({percent}): {t:.3f}")
    return t


def extract_true_neg_candidate(csv):
    
    thres_percentile = 5
    df_train_neg = pd.read_csv(csv)
    df_0 = df_train_neg[df_train_neg["olabel"]==0]
    df_1 = df_train_neg[df_train_neg["olabel"]==1]
    df_2 = df_train_neg[df_train_neg["olabel"]==2]
    df_neg = df_train_neg[df_train_neg["olabel"]==-1]
    
    p1=percentile(df_0["prob_0"], thres_percentile)
    p2=percentile(df_1["prob_1"], thres_percentile)
    p3=percentile(df_2["prob_2"], thres_percentile)
    th = np.min([p1,p2,p3])
    
    df_neg_true = df_neg[np.max(df_neg[["prob_0","prob_1","prob_2"]], axis=1) < th]
    return df_neg_true


def extract_neg_index(model_dir):
    df_neg_true_0= extract_true_neg_candidate(os.path.join(model_dir, "train_neg_0.csv"))
    df_neg_true_1= extract_true_neg_candidate(os.path.join(model_dir, "train_neg_1.csv"))
    df_neg_true_2= extract_true_neg_candidate(os.path.join(model_dir, "train_neg_2.csv"))
    
    neg_index_0 = df_neg_true_0["orig_index"].values.tolist()
    neg_index_1 = df_neg_true_1["orig_index"].values.tolist()
    neg_index_2 = df_neg_true_2["orig_index"].values.tolist()
    
    true_neg_indexes = sorted(list(set(neg_index_0 ).intersection(set(neg_index_1 )).intersection(set(neg_index_2))))
    return true_neg_indexes




