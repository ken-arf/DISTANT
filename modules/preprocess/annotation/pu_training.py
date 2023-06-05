import pandas as pd
import spacy
import scispacy
import os
import time
import pickle
import numpy as np
import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import get_scheduler
from accelerate import Accelerator
import evaluate


from tqdm.auto import tqdm

from utils import utils

from pu_dataloader import PU_Dataloader 
#from pu_model import PU_Model 
from pu_model2 import PU_Model 
#from pu_model3 import PU_Model 
#from pu_samples import extract_neg_index
from pu_samples import classify_unknown_samples

from pu_samples import generate_final_pos_samples

from measure import performance
import pdb

pd.set_option('display.max_colwidth', None)


def train(df_train_pos, df_train_neg, parameters, model_name_suffix):

    # logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.ERROR)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

    #handler2 = logging.FileHandler(filename="test.log")
    #handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(handler1)
    #logger.addHandler(handler2)

    pu_dataloader = PU_Dataloader(df_train_pos, df_train_neg, parameters)

    train_dataloader, valid_dataloader, test_dataloader = pu_dataloader.load_data()

    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    pu_model = PU_Model(parameters, logger)

    if parameters['restore_model'] == True:
        model_path = parameters['restore_model_path'].replace("%suffix%", model_name_suffix)
        pu_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    optimizer = AdamW(pu_model.parameters(), lr=float(parameters['train_lr']))
    accelerator = Accelerator()
    pu_model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        pu_model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )

    num_train_epochs = parameters['train_epochs']
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    print("num_update_steps_per_epoch", num_update_steps_per_epoch)
    print("num_train_epochs", num_train_epochs)
    print("num_training_steps", num_training_steps)

    lr_scheduler = get_scheduler(
        parameters['train_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=parameters['train_num_warmup_steps'],
        num_training_steps=num_training_steps,
    )

    # training model

    progress_bar = tqdm(range(num_training_steps))

    OUTPUT_PATH = parameters['model_dir']
    best_acc = 0
    patient_count = 0
    steps = 0

    for epoch in range(num_train_epochs):

        # Training
        pu_model.train()
        for batch in train_dataloader:

            loss = pu_model(**batch)
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description("loss:{:7.2f} epoch:{}".format(loss.item(),epoch))
            steps += 1


        # Evaluation

        progress_bar_valid = tqdm(range(len(valid_dataloader)))
        running_acc = 0
        pu_model.eval()
        for batch_index, batch in tqdm(enumerate(valid_dataloader)):
            with torch.no_grad():
                predictions, probs = pu_model.decode(**batch)
            labels = batch["labels"].detach().cpu().numpy()
            batch_acc = performance.performance_acc(predictions, labels, logger)
            running_acc += (batch_acc - running_acc) / (batch_index + 1)
            progress_bar_valid.update(1)
            progress_bar_valid.set_description("running_acc:{:.2f} epoch:{}".format(running_acc, epoch))
        print("running_acc: {:.2f}".format(running_acc))


        best_update = False
        if best_acc < running_acc:
            best_acc = running_acc
            # save model
            patient_count = 0
            best_update = True
        else:
            patient_count += 1

        if patient_count > parameters["max_patient_count"]:
            print("Exceed max patient count. Force to exit")
            break

        # prepare saving model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(pu_model)

        OUTPUT_PATH = parameters['model_dir']

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_PATH, f"model_last_{model_name_suffix}.pth"))

        if best_update:
            torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_PATH, f"model_best_{model_name_suffix}.pth"))

    print("best_acc: {:.2f}".format(best_acc))

    if test_dataloader == None:
        return

    # load best model
    best_model_path = os.path.join(OUTPUT_PATH, f"model_best_{model_name_suffix}.pth")
    pu_model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))
    #pu_model.load_state_dict(torch.load(parameters['restore_model_path'], map_location=torch.device(device)))

    # testing
    progress_bar_test = tqdm(range(len(test_dataloader)))
    probs_list = []
    pu_model.eval()
    for batch_index, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            predictions, probs = pu_model.decode(**batch)
        labels = batch["labels"].detach().cpu().numpy()
        batch_acc = performance.performance_acc(predictions, labels, logger)
        progress_bar_test.update(1)
        progress_bar_test.set_description("acc:{:.2f}".format(batch_acc))
        probs_list.append(probs)

    neg_probs = np.vstack(probs_list)

    return neg_probs


def setup_pu_dataset(step, parameters):

    #frac_spy = 0.05
    #frac_spy = 0.1
    frac_spy = parameters["spy_ratio"]


    corpus_dir = parameters["corpus_dir"]

    seed = parameters["seed"] + step

    print("load dataset", os.path.join(corpus_dir,"df_train_pos_neg.csv"))
    df_train_raw = pd.read_csv(os.path.join(corpus_dir,"df_train_pos_neg.csv"), index_col=0)


    #df_train_raw = df_train_raw.dropna()
    df_train_raw.reset_index(drop=True, inplace=True)
    df_train_raw["orig_index"] = df_train_raw.index

    label_types = list(df_train_raw["label"].unique())

    df_unknown = df_train_raw[df_train_raw["label"]==-1]

    # ignore label == -1
    num_pos_labels = len(label_types) - 1
    df_trains = {}
    for k in range(num_pos_labels):
        df_trains[k] = df_train_raw[df_train_raw["label"] == k]


    df_train_spys = {}
    for k in range(num_pos_labels):
        df_train_spys[k] = df_trains[k].sample(frac=frac_spy, random_state=seed)


    df_train_diffs = {}
    for k in range(num_pos_labels):
        indexes = [i for i in df_trains[k].index.tolist() if i not in df_train_spys[k].index.tolist()]
        df_train_diffs[k] = df_trains[k].loc[indexes]

    
    df_train_pos_step = pd.concat(list(df_train_diffs.values()))
    df_train_neg_step = pd.concat([df_unknown] + list(df_train_spys.values()))


    print("df_train_raw", df_train_raw.shape)
    print("df_train_pos_step", df_train_pos_step.shape)
    print("df_train_neg_step", df_train_neg_step.shape)

    df_train_neg_step.rename(columns={"label": "olabel"}, inplace=True)
    labels = [-1] * df_train_neg_step.shape[0]
    df_train_neg_step["label"] = labels

    df_train_pos_step["olabel"] = df_train_pos_step.loc[:,"label"]

    return df_train_pos_step, df_train_neg_step

def setup_final_dataset(unknown_sample_result, parameters):


    corpus_dir = parameters["corpus_dir"]
    df_train_raw = pd.read_csv(os.path.join(corpus_dir,"df_train_pos_neg.csv"), index_col=0)

    ## for debug ####
    #df_train_raw = df_train_raw.iloc[:200]
    #################

    #df_train_raw = df_train_raw.dropna()
    df_train_raw.reset_index(drop=True, inplace=True)
    df_train_raw["orig_index"] = df_train_raw.index
    df_train_raw["olabel"] = df_train_raw.label


    pos_labels = [l for l in df_train_raw["label"].unique() if l >= 0]
    neg_label = len(pos_labels)

    # negative unknown samples
    neg_index = unknown_sample_result['neg_index']
    df_train_raw.loc[neg_index, "label"] = neg_label

    # positive unknown samples
    pos_index = unknown_sample_result['pos_index']
    pos_label = unknown_sample_result['pos_label']
    df_train_raw.loc[pos_index, "label"] = pos_label

    df_train = df_train_raw[df_train_raw.label >=0]
    # shuffle training dataset
    df_train = df_train.sample(frac=1)
 
    return df_train


def extract_negative_samples(parameters, count):

    pos_label_num = parameters["class_num"]

    for step in range(count):

        parameters["seed"] = step
        df_train_pos, df_train_neg = setup_pu_dataset(step, parameters)
        neg_probs = train(df_train_pos, df_train_neg, parameters, model_name_suffix="pu")

        if pos_label_num > 1:
            for k in range(pos_label_num):
                df_train_neg[f"prob_{k}"] = neg_probs[:, k]
        else:
            df_train_neg[f"prob_0"] = 1.0 - neg_probs[:, 0]


        model_dir = parameters["model_dir"]
        df_train_pos.to_csv(os.path.join(model_dir, f"train_pos_{step}.csv"))
        df_train_neg.to_csv(os.path.join(model_dir, f"train_neg_{step}.csv"))

    return

def retrain_classifier(unknown_sample_result, parameters):

    # add negative label class 
    parameters["class_num"] += 1
    df_train = setup_final_dataset(unknown_sample_result, parameters)

    print("retrain_classifier with unknown samples")
    train(df_train, None, parameters, model_name_suffix="final")

    model_dir = parameters["model_dir"]

    df_train.set_index('orig_index')
    df_train.to_csv(os.path.join(model_dir, "train.csv"))


def main():

    # set config path by command line
    inp_args = utils._parsing()                                                                                            
    config_path = getattr(inp_args, 'yaml')
    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    # print config
    utils._print_config(parameters, config_path)

    # check running time
    t_start = time.time()                                                                                                  

    # step-1.
    # extract (true) negative samples
    # use PU-algorithm to extract possibly true negative samples from unknown samples
    # by using spy positive samples. We iterate this process three timees, then extract
    # negative samples by the common sample set from the three sets of negative samples
    

    count = 3
    extract_negative_samples(parameters, count=count)
    model_dir = parameters["model_dir"]
    unknow_sample_result = classify_unknown_samples(model_dir, parameters["pu_thres_pert"], count=count)
    
    # step-2.
    # classify unknown samples
    # By using the positive and the negative samples extractd in step 1, we train a NN classifier
    # to classify each unknown sample whether it is one of the possitive classes or negative class
    retrain_classifier(unknow_sample_result, parameters)

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':                                                                                                                        
    main()



