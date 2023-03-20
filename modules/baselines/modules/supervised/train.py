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

from utils import utils

import torch
from torch.optim import AdamW

from transformers import get_scheduler

from accelerate import Accelerator
import evaluate

from dataloader import Dataloader
from measure import performance
#from model import Model
from model_crf import Model


import pdb


def load_file(file):
    
    input_data = defaultdict(list)
    
    with open(file) as fp:
        tokens = []
        bio_labels = []
        
        for line in fp:
            line = line.strip('\n')
            #print(line, len(line))
            if len(line) == 0:
                input_data['tokens'].append(tokens)
                input_data['bio'].append(bio_labels)
                tokens = []
                bio_labels = []
                continue
            fields = line.split('\t')
            #print(fields)
            assert(len(fields) == 2)
            tokens.append(fields[0].lower())
            bio_labels.append(fields[1])
            
    return input_data
            

def convert_bio(ent2id, label):

    label = label.replace("S-", "B-")
    return ent2id[label]

def load_dataset(data_kind, parameters):

    #pdb.set_trace()

    corpus_dir = parameters["corpus_dir"][data_kind]

    files = sorted(glob.glob(f"{corpus_dir}/*.coll"))

    text_data = []
    label_data = []

    for file in tqdm(files):
        #print(file)
        input_data = load_file(file)
        #print(input_data.keys())
        seq_num = len(input_data['tokens'])
        
        for sn in range(seq_num):

            text_data.append(input_data['tokens'][sn])
            b_bio = [parameters['ent2id'][bio.replace('S-', 'B-')] for bio in input_data['bio'][sn]]
            label_data.append(b_bio)

    data = {'text': text_data, 'label': label_data}

    return data

def train(parameters, name_suffix):

    # logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.ERROR)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

    #handler2 = logging.FileHandler(filename="test.log")
    #handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(handler1)
    #logger.addHandler(handler2)

    ent2id = {}
    ent2id['O'] = 0
    for entity in parameters["entity_names"]:
        ent2id[f'B-{entity}'] = len(ent2id)
        ent2id[f'I-{entity}'] = len(ent2id)
    id2ent = {d:k for k, d in ent2id.items()}

    parameters["ent2id"] = ent2id
    parameters["id2ent"] = id2ent

    # step 0) load dataset
    train_data = load_dataset('train', parameters)
    dev_data = load_dataset('dev', parameters)
    test_data = load_dataset('test', parameters)

    dataset = {'train': train_data, 'dev': dev_data, 'test': test_data}
    dataloader = Dataloader(dataset, parameters)
    train_dataloader, valid_dataloader, test_dataloader = dataloader.load_data()

    print("Train data loader size: ", len(train_dataloader))
    print("Valid data loader size: ", len(valid_dataloader))
    print("Test data loader size: ", len(test_dataloader))

    metric = evaluate.load("seqeval")

    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    model = Model(parameters, logger)

    if parameters['restore_model'] == True:
        model.load_state_dict(torch.load(parameters['restore_model_path'], map_location=torch.device(device)))

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
    best_acc = 0
    patient_count = 0
    steps = 0
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
            progress_bar.set_description("loss:{:7.2f} epoch:{}".format(loss.item(),epoch))
            steps += 1

            ## debug
            break

        # Evaluation
        progress_bar_valid = tqdm(range(len(valid_dataloader)))
        running_acc = 0
        model.eval()
        for batch_index, batch in tqdm(enumerate(valid_dataloader)):
            with torch.no_grad():
                predictions = model.decode(**batch)
            labels = batch["labels"].detach().cpu().numpy()


            #predictions = accelerator.pad_across_processes(predictions, dim=2, pad_index=-100)
            #labels = accelerator.pad_across_processes(labels, dim=2, pad_index=-100)

            predictions = accelerator.gather(predictions)
            labels = accelerator.gather(labels)

            logger.debug("predictions")
            logger.debug(predictions)
            logger.debug("labels")
            logger.debug(labels)

            true_predictions, true_labels = utils.postprocess(predictions, labels, id2ent)

            batch_acc = performance.performance_acc(true_predictions, true_labels, logger)
            running_acc += (batch_acc - running_acc) / (batch_index + 1)
            progress_bar_valid.update(1)
            progress_bar_valid.set_description("running_acc:{:.2f} epoch:{}".format(running_acc, epoch))

            ## debug
            #break

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
        
        ## debug 
        #best_update = True
        ####

        # prepare saving model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        OUTPUT_PATH = parameters['model_dir']

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_PATH, f"model_last_{name_suffix}.pth"))

        if best_update:
            torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_PATH, f"model_best_{name_suffix}.pth"))

        ## debug
        #break

    print("best_acc: {:.2f}".format(best_acc))

    # test
    #test(logger, parameters, test_dataloader, name_suffix)

    return

def test(logger, parameters, test_dataloader, name_suffix):

    print("test start")

    ent2id = {}
    ent2id['O'] = 0
    for entity in parameters["entity_names"]:
        ent2id[f'B-{entity}'] = len(ent2id)
        ent2id[f'I-{entity}'] = len(ent2id)
    id2ent = {d:k for k, d in ent2id.items()}

    parameters["ent2id"] = ent2id
 
    metric = evaluate.load("seqeval")
    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    OUTPUT_PATH = parameters['model_dir']
    model = Model(parameters, logger)
    best_model_path = os.path.join(OUTPUT_PATH, f"model_best_{name_suffix}.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))

    accelerator = Accelerator()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # test
    progress_bar_test = tqdm(range(len(test_dataloader)))
    running_acc = 0
    model.eval()
    for batch_index, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            predictions, probs = model.decode(**batch)
        labels = batch["labels"].detach().cpu().numpy()

        predictions = accelerator.gather(predictions)
        labels = accelerator.gather(labels)

        logger.debug("predictions")
        logger.debug(predictions)
        logger.debug("labels")
        logger.debug(labels)

        true_predictions, true_labels = utils.postprocess(predictions, labels, id2ent)

        batch_acc = performance.performance_acc(true_predictions, true_labels, logger)
        running_acc += (batch_acc - running_acc) / (batch_index + 1)
        progress_bar_test.update(1)
        progress_bar_test.set_description("running_acc:{:.2f}".format(running_acc))


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

    # train model
    train(parameters, "supervised")

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))
        

if __name__ == '__main__':                                                                                                                        
    main()


