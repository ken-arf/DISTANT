#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import torch
import logging
import glob
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from utils import utils

import torch
from torch.optim import AdamW

from transformers import get_scheduler

from accelerate import Accelerator
import evaluate

from dataloader_weight import Dataloader
from measure import performance
from model_weight import Model

import pdb

def load_file(file):

    input_data = defaultdict(list)

    with open(file) as fp:
        tokens = []
        label_weights = []
        bio_labels = []

        for line in fp:
            line = line.strip('\n')
            # print(line, len(line))
            if len(line) == 0:
                input_data['tokens'].append(tokens)
                input_data['weight'].append(label_weights)
                input_data[f'bio'].append(bio_labels)
                tokens = []
                label_weights = []
                bio_labels = []
                continue
            fields = line.split('\t')
            tokens.append(fields[0])
            label_weights.append(int(fields[-2]))
            bio_labels.append(fields[-1])

    return input_data


def load_dataset(parameters):

    label2int = {'O': 0, 'B': 1, 'I': 2, 'S':3}

    ent_num = parameters["num_bio_labels"]
    conll_dir = parameters["conll_dir"]
    finetune_dir = parameters["finetune_dir"]

    files = sorted(glob.glob(f"{conll_dir}/*.txt"))
    

    text_data = []
    weight_data = []
    label_data = []


    for file in tqdm(files):
        _, fname = os.path.split(file)
        base, _  = os.path.splitext(fname)

        # print(file)
        input_data = load_file(file)
        # print(input_data.keys())
        seq_num = len(input_data['tokens'])

        for sn in range(seq_num):

            text_data.append(input_data['tokens'][sn])
            # map class name to integer
            label_seq = list(map(lambda x: label2int[x], input_data['bio'][sn]))
            label_data.append(label_seq)

            weight_data.append(input_data['weight'][sn])
    
    # split val data in halves
    n = int(len(text_data) * 0.9)

    data = {
            'train': {'text': text_data[:n], 'label': label_data[:n], 'weight': weight_data[:n]},
            'val': {'text': text_data[n:], 'label': label_data[n:], 'weight': weight_data[n:]}
            }

    return data


def train_finetune(parameters, name_suffix):
    

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

    # step 0) load dataset
    data = load_dataset(parameters)
    dataloader = Dataloader(data, parameters)
    # train_dataloader = dataloader.load_data_train_only()
    train_dataloader, val_dataloader = dataloader.load_data_preset()

    print("train data loader size: ", len(train_dataloader))
    print("val data loader size: ", len(train_dataloader))


    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
    else:
        device = torch.device("mps")


    device = torch.device("cpu")

    model = Model(parameters, logger)
    model = model.to(device)

    if parameters['restore_model'] == True:
        model.load_state_dict(torch.load(
            parameters['restore_model_path'], map_location=device))


    optimizer = AdamW(model.parameters(), lr=float(parameters['train_lr']))

    #accelerator = Accelerator()
    #model, optimizer, train_dataloader = accelerator.prepare(
    #    model, optimizer, train_dataloader
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

    model.train()
    steps = 0
    for epoch in range(num_train_epochs):

        # Training
        for batch in train_dataloader:

            loss = model(**batch)
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
    #unwrapped_model = accelerator.unwrap_model(model)

    OUTPUT_PATH = parameters['model_dir']

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    #torch.save(unwrapped_model.state_dict(), os.path.join(
    #    OUTPUT_PATH, f"model_finetune_{name_suffix}.pth"))

    torch.save(model.state_dict(), os.path.join(
        OUTPUT_PATH, f"model_segmentation.pth"))

def train(parameters, name_suffix):


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

    # step 0) load dataset
    data = load_dataset(parameters)
    dataloader = Dataloader(data, parameters)


    train_dataloader, valid_dataloader = dataloader.load_data_preset()

    print("Train data loader size: ", len(train_dataloader))
    print("Valid data loader size: ", len(valid_dataloader))

    #metric = evaluate.load("seqeval")

    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    model = Model(parameters, logger)

    if parameters['restore_model'] == True:

        if parameters['cnt'] == 1:
            model.load_state_dict(torch.load(
                parameters['restore_model_base_path'], map_location=torch.device(device)))
        else:
            model.load_state_dict(torch.load(
                parameters['restore_model_prev_path'], map_location=torch.device(device)))


    optimizer = AdamW(model.parameters(), lr=float(parameters['train_lr']))
    accelerator = Accelerator()
    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader
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
    patient_count = 0
    steps = 0

    best_acc = 0
  
    for epoch in range(num_train_epochs):

        # Training
        model.train()
        for batch in train_dataloader:

            loss = model(**batch)
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
        model.eval()
        for batch_index, batch in tqdm(enumerate(valid_dataloader)):
            with torch.no_grad():
                predictions, probs = model.decode(**batch)
            labels = batch["labels"].detach().cpu().numpy()

            # predictions = accelerator.pad_across_processes(predictions, dim=2, pad_index=-100)
            # labels = accelerator.pad_across_processes(labels, dim=2, pad_index=-100)

            predictions = accelerator.gather(predictions)
            labels = accelerator.gather(labels)

            logger.debug("predictions")
            logger.debug(predictions)
            logger.debug("labels")
            logger.debug(labels)

            label_map = {0: 0, 1: 1, 2: 2, 3: 3}
            true_predictions, true_labels = utils.postprocess(
                predictions, labels, label_map)

            batch_acc = performance.performance_acc(
                true_predictions, true_labels, logger)
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
        unwrapped_model = accelerator.unwrap_model(model)

        OUTPUT_PATH = parameters['model_dir']

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

#        torch.save(unwrapped_model.state_dict(), os.path.join(
#            OUTPUT_PATH, f"model_last_{name_suffix}.pth"))

        if best_update:
            torch.save(unwrapped_model.state_dict(), os.path.join(
                OUTPUT_PATH, f"model_segmentation.pth"))

    print("best_acc: {:.2f}".format(best_acc))
    return


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

    # train model
    #train_finetune(parameters, "span")
    train(parameters, "span")

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()
