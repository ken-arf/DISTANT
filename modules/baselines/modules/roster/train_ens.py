#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import torch
import logging
import glob
import numpy as np
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
from model import RoSTER, RoSTER_ENS

import pdb


def load_file(file,  n_type):

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
            assert(len(fields) == 3)
            tokens.append(fields[0])
            bio_labels.append(fields[2])
            
    return input_data
            

def load_dataset(parameters):

    ent_num = parameters["num_bio_labels"]
    corpus_dir = parameters["corpus_dir"]

    files = sorted(glob.glob(f"{corpus_dir}/*.txt"))

    text_data = []
    bio_labels = []


    for file in tqdm(files):
        input_data = load_file(file, ent_num)
        text_data += input_data['tokens']
        bio_labels += input_data['bio']


    data = {'text': text_data, 
            'bio_labels': bio_labels}

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


    label2int = parameters["label2int"]
    int2label = {v:k for k, v in label2int.items()}

    data = load_dataset(parameters)
    dataloader = Dataloader(data, parameters, logger)
    train_dataloader, valid_dataloader = dataloader.load_data()

    print("Train data loader size: ", len(train_dataloader))
    print("Valid data loader size: ", len(valid_dataloader))


    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    # RoSTER model
    models = [RoSTER(parameters, logger) for i in range(5)]

    # RoSTER_ENS model
    model_ens = RoSTER_ENS(parameters, logger)

    if parameters['restore_model'] == True:
        for model, weight in zip(models, sorted(glob.glob(parameters['restore_model_path']))):
            model.load_state_dict(torch.load(weight, map_location=torch.device(device)))

    optimizer = AdamW(model_ens.parameters(), lr=float(parameters['train_lr']))
    accelerator = Accelerator()
    model_ens, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        model_ens, optimizer, train_dataloader, valid_dataloader
    )

    #clipping_value = 5 # arbitrary value of your choosing
    #torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)

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
        model_ens.train()
        for model in models:
            model.eval()

        for batch in train_dataloader:

            with torch.no_grad():
                all_probs = []
                for model in models:
                    predictions, probs = model.decode(**batch)
                    all_probs.append(np.expand_dims(probs, axis=0))
                probs_mean = np.mean(np.concatenate(all_probs, axis=0), axis=0)


            batch.update({"softlabels":probs})
            probs, loss = model_ens(**batch)
            accelerator.backward(loss)
            #loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description("loss:{:7.2f} epoch:{}".format(loss.item(),epoch))
            steps += 1
            break


        # Evaluation
        progress_bar_valid = tqdm(range(len(valid_dataloader)))

        running_acc = 0

        model_ens.eval()
        for batch_index, batch in tqdm(enumerate(valid_dataloader)):
            with torch.no_grad():
                predictions, probs = model_ens.decode(**batch)
            labels = batch["labels"].detach().cpu().numpy()


            #predictions = accelerator.pad_across_processes(predictions, dim=2, pad_index=-100)
            #labels = accelerator.pad_across_processes(labels, dim=2, pad_index=-100)

            predictions = accelerator.gather(predictions)
            labels = accelerator.gather(labels)

            logger.debug("predictions")
            logger.debug(predictions)
            logger.debug("labels")
            logger.debug(labels)

            true_predictions, true_labels = utils.postprocess(predictions, labels, int2label)

            batch_acc = performance.performance_acc(true_predictions, true_labels, logger)
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
        unwrapped_model = accelerator.unwrap_model(model_ens)

        OUTPUT_PATH = parameters['model_dir']

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_PATH, f"model_last_{name_suffix}.pth"))

        if best_update:
            torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_PATH, f"model_best_{name_suffix}.pth"))

    print("best_acc: {:.2f}".format(best_acc))
    return


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
    train(parameters, "roster_ens")

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))
        

if __name__ == '__main__':                                                                                                                        
    main()


