import pandas as pd
import spacy
import scispacy
import os
import time
import pickle
import numpy as np
import glob
from datetime import datetime
import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import get_scheduler
from accelerate import Accelerator

from tqdm.auto import tqdm

from utils import utils
# from utils import database

from pu_dataloader import PU_Dataloader
from pu_model import PU_Model

from measure import performance

import pdb

pd.set_option('display.max_colwidth', None)

def finetune(df_train_pos, parameters, model_name_suffix):

    # logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.ERROR)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter(
        "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

    # handler2 = logging.FileHandler(filename="test.log")
    # handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(handler1)
    # logger.addHandler(handler2)

    pu_dataloader = PU_Dataloader(df_train_pos, None, parameters)

    train_dataloader = pu_dataloader.load_train_only()

    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    pu_model = PU_Model(parameters, logger)


    model_state = pu_model.state_dict()

    if parameters['restore_model'] == True:
        model_path = parameters['restore_model_path'].replace(
            "%suffix%", model_name_suffix)

        pretrained_state = torch.load(model_path, map_location=torch.device(device))
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
        model_state.update(pretrained_state)
        pu_model.load_state_dict(model_state)

        #pu_model.load_state_dict(torch.load(
        #    model_path, map_location=torch.device(device)))

    optimizer = AdamW(pu_model.parameters(), lr=float(parameters['train_lr']))
    #accelerator = Accelerator()
    #pu_model, optimizer, train_dataloader = accelerator.prepare(
    #    pu_model, optimizer, train_dataloader
    #)

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
    steps = 0

    for epoch in range(num_train_epochs):


        # Training
        pu_model.train()
        for batch in train_dataloader:

            loss = pu_model(**batch)
            #accelerator.backward(loss)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(
                "loss:{:7.2f} epoch:{}".format(loss.item(), epoch))
            steps += 1

    # prepare saving model
    #accelerator.wait_for_everyone()
    #unwrapped_model = accelerator.unwrap_model(pu_model)

    OUTPUT_PATH = parameters['model_dir']

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    #torch.save(unwrapped_model.state_dict(), os.path.join(
    #    OUTPUT_PATH, f"model_best_{model_name_suffix}.pth"))

    torch.save(pu_model.state_dict(), os.path.join(
        OUTPUT_PATH, f"model_spanClassification.pth"))

    return


def train(df_train_pos, df_train_neg, parameters, model_name_suffix):

    # logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.ERROR)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter(
        "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

    # handler2 = logging.FileHandler(filename="test.log")
    # handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(handler1)
    # logger.addHandler(handler2)

    pu_dataloader = PU_Dataloader(df_train_pos, df_train_neg, parameters)

    #train_dataloader, valid_dataloader, test_dataloader = pu_dataloader.load_data()
    train_dataloader, valid_dataloader, test_dataloader = pu_dataloader.load_data_preset()

    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    pu_model = PU_Model(parameters, logger)
    model_state = pu_model.state_dict()

    if parameters['restore_model'] == True:
        model_path = parameters['restore_model_path'].replace(
            "%suffix%", model_name_suffix)

        pretrained_state = torch.load(model_path, map_location=torch.device(device))
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
        model_state.update(pretrained_state)
        pu_model.load_state_dict(model_state)


#    if parameters['restore_model'] == True:
#        model_path = parameters['restore_model_path'].replace(
#            "%suffix%", model_name_suffix)
#        pu_model.load_state_dict(torch.load(
#            model_path, map_location=torch.device(device)))

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
            progress_bar.set_description(
                "loss:{:7.2f} epoch:{}".format(loss.item(), epoch))
            steps += 1

        # Evaluation

        progress_bar_valid = tqdm(range(len(valid_dataloader)))
        running_acc = 0
        pu_model.eval()
        for batch_index, batch in tqdm(enumerate(valid_dataloader)):
            with torch.no_grad():
                predictions, probs = pu_model.decode(**batch)
            labels = batch["labels"].detach().cpu().numpy()
            batch_acc = performance.performance_acc(
                predictions, labels, logger)
            running_acc += (batch_acc - running_acc) / (batch_index + 1)
            progress_bar_valid.update(1)
            progress_bar_valid.set_description(
                "running_acc:{:.2f} epoch:{}".format(running_acc, epoch))
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

        #torch.save(unwrapped_model.state_dict(), os.path.join(
        #    OUTPUT_PATH, f"model_last_{model_name_suffix}.pth"))

        if best_update:
            torch.save(unwrapped_model.state_dict(), os.path.join(
                OUTPUT_PATH, f"model_classification.pth"))

    print("best_acc: {:.2f}".format(best_acc))

    if test_dataloader == None:
        return

    # load best model
    best_model_path = os.path.join(
        OUTPUT_PATH, f"model_best_{model_name_suffix}.pth")
    pu_model.load_state_dict(torch.load(
        best_model_path, map_location=torch.device(device)))
    # pu_model.load_state_dict(torch.load(parameters['restore_model_path'], map_location=torch.device(device)))

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


def setup_final_dataset( parameters):


    document_root = parameters["corpus_dir"]
    finetune_dir = parameters["finetune_dir"]

#    with open(os.path.join(finetune_dir, 'updated_doc.pmid')) as fp:
#        train_pmids = [pmid.strip() for pmid in fp.readlines()]
#
#    with open(os.path.join(finetune_dir, 'unchanged_doc.pmid')) as fp:
#        val_pmids = [pmid.strip() for pmid in fp.readlines()]
    
    files = glob.glob(f"{document_root}/*.csv")

    dfs = []

    for file in files:
        print(file)
        _, fname = os.path.split(file)
        base, _ = os.path.splitext(fname)

        df = pd.read_csv(file, index_col = 0)
        print(df)
        dfs.append(df)


    n = int(len(dfs) * 0.9)

    df_train = pd.concat(dfs[:n]).reset_index(drop=True)
    df_val = pd.concat(dfs[n:]).reset_index(drop=True)

    return df_train, df_val


def finetune_classifier(parameters):

    # add negative label class
    df_train, df_val  = setup_final_dataset(parameters)
   
    df_train['split'] = 'train'
    df_val['split'] = 'val'

    df = pd.concat([df_train, df_val])


    # save training dataset for final classifier
    model_dir = parameters["model_dir"]
    utils.makedir(model_dir)

    #df_train.to_csv(os.path.join(model_dir, "train_dataset.csv"))
    df.to_csv(os.path.join(model_dir, "train_dataset.csv"))

    # shuffle training data
    df = df.sample(frac=1)

    # train final classifier
    print("finetune classifier")
    #finetune(df_train, parameters, model_name_suffix="classifier")
    train(df, None, parameters, model_name_suffix="classifier")


def main():

    # set config path by command line
    #inp_args = utils._parsing_timestamp()
    inp_args = utils._parsing()

    config_path = getattr(inp_args, 'yaml')
    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    #timestamp = getattr(inp_args, 'timestamp')
    #model_dir = parameters['model_dir']
    #model_dir = model_dir.replace('{timestamp}', timestamp)
    #parameters['model_dir'] = model_dir

    # print config
    utils._print_config(parameters, config_path)

    # check running time
    t_start = time.time()

    #entity_types = database.get_entity_types(parameters)
    entity_types = parameters['entity_names']
    parameters['class_num'] = len(entity_types) + 1

    finetune_classifier(parameters)

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()