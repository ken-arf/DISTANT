

import os
import pickle
import numpy as np
import pandas as pd

import pdb


def percentile(prob, percent):
    t = np.percentile(prob, percent)
    # print(f"percentile ({percent}): {t:.3f}")
    return t


def extract_unknown_samples(csv, thres_pert):


    prob_thres = 0.90

    thres_percentile = thres_pert
    df_train_neg = pd.read_csv(csv)
    df_neg = df_train_neg[df_train_neg["olabel"] == -1]

    pos_label_num = len(df_train_neg["olabel"].unique()) - 1

    # spy data
    dfs = {}
    for k in range(pos_label_num):
        dfs[k] = df_train_neg[df_train_neg["olabel"] == k]

    probs = {}
    for k in dfs.keys():
        p_max = np.max(dfs[k][f"prob_{k}"])
        p_min = np.min(dfs[k][f"prob_{k}"])
        p_mean = np.mean(dfs[k][f"prob_{k}"])
        p_per_low = percentile(dfs[k][f"prob_{k}"], thres_percentile)
        p_per_high = percentile(dfs[k][f"prob_{k}"], 100 - thres_percentile)
        probs[k] = {'max': p_max, 'min': p_min, 'mean': p_mean, 'per_low': p_per_low, 'per_high': p_per_high}

    print("probs")
    for key in probs.keys():
        print(probs[key])

    prob_col_names = [f"prob_{k}" for k in dfs.keys()]
    max_prob = np.max(df_neg[prob_col_names].values, axis=1)
    max_index = np.argmax(df_neg[prob_col_names].values, axis=1)

    df_neg["max_prob"] = max_prob
    df_neg["max_index"] = max_index

    #changed 2023/10/14
    thres_pos = [probs[index]['per_high'] for index in max_index]
    thres_neg = [probs[index]['per_low'] for index in max_index]

    mask = max_prob < np.array(thres_neg)
    df_neg_unknown = df_neg[mask]

    mask = max_prob >= np.array(thres_pos)
    df_pos_unknown = df_neg[mask]
    pos_label = max_index[mask]

    cols = ["max_prob", "max_index", "orig_index", "label"]
    df_pos_unknown_ = df_pos_unknown[cols]
    df_neg_unknown_ = df_neg_unknown[cols]

    df_pos_unknown_.set_index('orig_index', inplace=True)
    df_neg_unknown_.set_index('orig_index', inplace=True)

    result = {'df_pos': df_pos_unknown_, 'df_neg': df_neg_unknown_}

    return result



def classify_unknown_samples(model_dir, thres_pert=5, count=1):


    candidate = {}
    for i in range(count):
        candidate[i] = extract_unknown_samples(
            os.path.join(model_dir, f"train_neg_{i}.csv"), thres_pert)

    # originally unknown samples which are judged as postive by PU training
    df_merge_pos = candidate[0]['df_pos']
    for i in range(1, count):
        df = candidate[i]['df_pos']
        df_temp = pd.merge(df_merge_pos, df, left_index=True, right_index=True)
        df_merge_pos = df_temp

    # originally unknown samples which are judged as negative by PU training
    df_merge_neg = candidate[0]['df_neg']
    for i in range(1, count):
        df = candidate[i]['df_neg']
        df_temp = pd.merge(df_merge_neg, df, left_index=True, right_index=True)
        df_merge_neg = df_temp

    cols_prob = [
        col for col in df_merge_pos.columns if col.startswith('max_prob')]
    cols_index = [
        col for col in df_merge_pos.columns if col.startswith('max_index')]

    pos_index = df_merge_pos.index.tolist()
    neg_index = df_merge_neg.index.tolist()
    pos_label = df_merge_pos[cols_index[0]].values.tolist()

    result = {'pos_index': pos_index,
              'neg_index': neg_index, 'pos_label': pos_label}

    return result



def generate_final_pos_samples(parameters):

    model_dir = parameters["model_dir"]

    # load pu training dataset
    df_pos = pd.read_csv(os.path.join(model_dir, "train_final.csv"))
    # load pu testing dataset
    df_neg = pd.read_csv(os.path.join(model_dir, "test_final.csv"))

    # extract prob information

    prob_cols = [col for col in list(df_neg.columns) if "prob_" in col]

    prob = df_neg[prob_cols].values
    # prob=df_neg[["prob_0","prob_1","prob_2","prob_3"]].values

    predict = np.argmax(prob, axis=1).tolist()
    df_neg["predict"] = predict

    #  take only positive results
    df_pos_inf = df_neg[df_neg["predict"] != len(prob_cols) - 1]
    # df_pos_inf=df_neg[df_neg["predict"] != 3]

    # aling colums names
    df_pos_inf.loc[:, "label"] = df_pos_inf.loc[:, "predict"]
    df_pos_inf = df_pos_inf.drop(prob_cols + ["predict"], axis=1)
    # df_pos_inf = df_pos_inf.drop(["prob_0","prob_1","prob_2","prob_3", "predict"], axis=1)

    # remove true negative samples from training data
    df_pos = df_pos[df_pos["label"] != len(prob_cols)-1]
    # df_pos = df_pos[df_pos["label"]!=3]

    # concat pos training and pos samples from inferences
    df_pu_train = pd.concat([df_pos, df_pos_inf], axis=0)

    columns = ["entities", "start_chars", "end_chars", "text", "pmid", "label"]
    df_pu_train = df_pu_train[columns]

    df_pu_train = df_pu_train.sort_values(by=['pmid'])

    df_pu_train['start_chars'] = df_pu_train['start_chars'].apply(np.int64)
    df_pu_train['end_chars'] = df_pu_train['end_chars'].apply(np.int64)

    return df_pu_train



if __name__ == "__main__":
    
    model_dir = "../../../experiments/immunology/models"
 
    classify_unknown_samples(model_dir,  thres_pert=5, count=3)
