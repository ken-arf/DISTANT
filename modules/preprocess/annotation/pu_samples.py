

import os
import pickle
import numpy as np
import pandas as pd



def percentile(prob, percent):
    t = np.percentile(prob, percent)
    #print(f"percentile ({percent}): {t:.3f}")
    return t


def extract_true_neg_candidate(csv, thres_pert):

    thres_percentile = thres_pert 
    df_train_neg = pd.read_csv(csv)
    df_neg = df_train_neg[df_train_neg["olabel"]==-1]

    pos_label_num = len(df_train_neg["olabel"].unique()) - 1

    # spy data
    dfs = {}
    for k in range(pos_label_num):
        dfs[k] = df_train_neg[df_train_neg["olabel"]==k]

    probs = {}
    for k in dfs.keys():
        p_max = np.max(dfs[k][f"prob_{k}"])
        p_min = np.min(dfs[k][f"prob_{k}"])
        p_mean = np.mean(dfs[k][f"prob_{k}"])
        p_per =percentile(dfs[k][f"prob_{k}"], thres_percentile)
        probs[k] = {'max': p_max, 'min': p_min, 'mean': p_mean, 'per': p_per}

    prob_col_names = [ f"prob_{k}" for k in dfs.keys() ]
    max_prob = np.max(df_neg[prob_col_names].values, axis=1)
    max_index = np.argmax(df_neg[prob_col_names].values, axis=1)
    #thres = [probs[index]['mean'] for index in max_index]
    thres = [probs[index]['per'] for index in max_index]
    
    mask = max_prob < np.array(thres)
    df_neg_true = df_neg[mask]

    return df_neg_true
    

def extract_true_neg_candidate_old(csv):
    
    thres_percentile = 5
    df_train_neg = pd.read_csv(csv)
    df_neg = df_train_neg[df_train_neg["olabel"]==-1]

    pos_label_num = len(df_train_neg["olabel"].unique()) - 1

    dfs = {}
    for k in range(pos_label_num):
        dfs[k] = df_train_neg[df_train_neg["olabel"]==k]
    
    #df_0 = df_train_neg[df_train_neg["olabel"]==0]
    #df_1 = df_train_neg[df_train_neg["olabel"]==1]
    #df_2 = df_train_neg[df_train_neg["olabel"]==2]
    
    probs = []
    for k in dfs.keys():
        print(dfs[k].shape)
        print("sampling label", k)
        if dfs[k].shape[0] == 0:
            continue

        p=percentile(dfs[k][f"prob_{k}"], thres_percentile)
        print(f"percentile at {thres_percentile} : {p}")
        probs.append(p)

    #p1=percentile(df_0["prob_0"], thres_percentile)
    #p2=percentile(df_1["prob_1"], thres_percentile)
    #p3=percentile(df_2["prob_2"], thres_percentile)

    th = np.min(probs)

    prob_col_names = [ f"prob_{k}" for k in dfs.keys() ]
    
    df_neg_true = df_neg[np.max(df_neg[prob_col_names], axis=1) < th]
    #df_neg_true = df_neg[np.max(df_neg[["prob_0","prob_1","prob_2"]], axis=1) < th]

    return df_neg_true

def extract_neg_index(model_dir, thres_pert = 15):

    # neg_sample result for 3 steps
    df_neg_true_0= extract_true_neg_candidate(os.path.join(model_dir, "train_neg_0.csv"), thres_pert)
    #df_neg_true_1= extract_true_neg_candidate(os.path.join(model_dir, "train_neg_1.csv"), thres_pert)
    #df_neg_true_2= extract_true_neg_candidate(os.path.join(model_dir, "train_neg_2.csv"), thres_pert)
    
    neg_index_0 = df_neg_true_0["orig_index"].values.tolist()
    #neg_index_1 = df_neg_true_1["orig_index"].values.tolist()
    #neg_index_2 = df_neg_true_2["orig_index"].values.tolist()
    
    #true_neg_indexes = sorted(list(set(neg_index_0 ).intersection(set(neg_index_1 )).intersection(set(neg_index_2))))
    return sorted(neg_index_0)

def extract_neg_index_old(model_dir, thres_pert = 15):

    # neg_sample result for 3 steps
    df_neg_true_0= extract_true_neg_candidate(os.path.join(model_dir, "train_neg_0.csv"), thres_pert)
    df_neg_true_1= extract_true_neg_candidate(os.path.join(model_dir, "train_neg_1.csv"), thres_pert)
    df_neg_true_2= extract_true_neg_candidate(os.path.join(model_dir, "train_neg_2.csv"), thres_pert)
    
    neg_index_0 = df_neg_true_0["orig_index"].values.tolist()
    neg_index_1 = df_neg_true_1["orig_index"].values.tolist()
    neg_index_2 = df_neg_true_2["orig_index"].values.tolist()
    
    true_neg_indexes = sorted(list(set(neg_index_0 ).intersection(set(neg_index_1 )).intersection(set(neg_index_2))))
    return true_neg_indexes


def generate_final_pos_samples(parameters):
    
    model_dir = parameters["model_dir"]
    
    # load pu training dataset
    df_pos = pd.read_csv(os.path.join(model_dir, "train_final.csv"))
    # load pu testing dataset
    df_neg = pd.read_csv(os.path.join(model_dir, "test_final.csv"))
    
    # extract prob information

    prob_cols = [ col for col in list(df_neg.columns) if "prob_" in col]

    prob=df_neg[prob_cols].values
    #prob=df_neg[["prob_0","prob_1","prob_2","prob_3"]].values

    predict=np.argmax(prob, axis=1).tolist()
    df_neg["predict"]=predict
    
    #  take only positive results
    df_pos_inf=df_neg[df_neg["predict"] != len(prob_cols) -1]
    #df_pos_inf=df_neg[df_neg["predict"] != 3]
    
    # aling colums names
    df_pos_inf.loc[:,"label"]=df_pos_inf.loc[:,"predict"]
    df_pos_inf = df_pos_inf.drop(prob_cols + ["predict"], axis=1)
    #df_pos_inf = df_pos_inf.drop(["prob_0","prob_1","prob_2","prob_3", "predict"], axis=1)
    
    # remove true negative samples from training data
    df_pos = df_pos[df_pos["label"]!= len(prob_cols)-1]
    #df_pos = df_pos[df_pos["label"]!=3]
    
    # concat pos training and pos samples from inferences
    df_pu_train = pd.concat([df_pos, df_pos_inf], axis=0)
    
    columns = ["entities","start_chars","end_chars","text","pmid","label"]
    df_pu_train = df_pu_train[columns]
    
    df_pu_train = df_pu_train.sort_values(by=['pmid'])
    
    df_pu_train['start_chars'] = df_pu_train['start_chars'].apply(np.int64)
    df_pu_train['end_chars'] = df_pu_train['end_chars'].apply(np.int64)
    
    return df_pu_train



