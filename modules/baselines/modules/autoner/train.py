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
from model_char_word import AutoNER
#from model_word import AutoNER

import pdb

def embedding(parameters):

    glove_embedding_path = parameters["glove_embeddig_path"]

    vocab,embeddings = [],[]
    with open(glove_embedding_path,'rt') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)

    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    print(vocab_npa[:10])

    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

    #insert embeddings for pad and unk tokens at top of embs_npa.
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
    print(embs_npa.shape)


def load_file(file,  n_type):

    input_data = defaultdict(list)
    
    with open(file) as fp:
        tokens = []
        bio_labels = defaultdict(list)
        spans = []
        
        for line in fp:
            line = line.strip('\n')
            #print(line, len(line))
            if len(line) == 0:
                input_data['tokens'].append(tokens)
                for i in range(n_type):
                    input_data[f'bio_{i}'].append(bio_labels[i])
                input_data['spans'].append(spans)
                tokens = []
                bio_labels = defaultdict(list)
                spans = []
                continue
            fields = line.split('\t')
            #print(fields)
            assert(len(fields) == n_type + 3)
            tokens.append(fields[0])
            for i in range(n_type):
                bio_labels[i].append(fields[2+i])
            spans.append(fields[-1])
            
    return input_data
            

def load_dataset(parameters):

    ent_num = parameters["num_bio_labels"]
    corpus_dir = parameters["corpus_dir"]

    files = sorted(glob.glob(f"{corpus_dir}/*.txt"))

    text_data = []
    bio_labels_0 = []
    bio_labels_1 = []
    bio_labels_2 = []
    spans = []


    for file in tqdm(files):
        input_data = load_file(file, ent_num)
        text_data += input_data['tokens']
        bio_labels_0 += input_data['bio_0']
        bio_labels_1 += input_data['bio_1']
        bio_labels_2 += input_data['bio_2']
        spans += input_data['spans']

    data = {'text': text_data, 
            'bio_labels_0': bio_labels_0,
            'bio_labels_1': bio_labels_1,
            'bio_labels_2': bio_labels_2,
            'spans': spans}

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

    # step 0) load dataset
    data = load_dataset(parameters)

    dataloader = Dataloader(data, parameters, logger)
    train_dataloader, valid_dataloader = dataloader.load_data()

    print("Train data loader size: ", len(train_dataloader))
    print("Valid data loader size: ", len(valid_dataloader))


    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    model = AutoNER(dataloader, parameters, logger)

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

            span_loss, ent_loss = model(**batch)
            loss = span_loss + ent_loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description("spanloss:{:7.2f} entloss:{:7.2f} epoch:{}".format(span_loss.item(),ent_loss.item(),epoch))
            steps += 1


        # Evaluation
        progress_bar_valid = tqdm(range(len(valid_dataloader)))

        running_acc = 0
        running_span_acc = 0
        running_entity_acc = 0

        model.eval()
        for batch_index, batch in tqdm(enumerate(valid_dataloader)):
            with torch.no_grad():
                span_result, entity_result = model.decode(**batch)


            logger.debug("Span")
            logger.debug("pred")
            logger.debug(span_result[0])
            logger.debug("true")
            logger.debug(span_result[1])

            logger.debug("Entity")
            logger.debug("pred")
            logger.debug(entity_result[0])
            logger.debug("true")
            logger.debug(entity_result[1])

            label_map = {0:'T', 1:'B'}
            pred_span, true_span = utils.postprocess(span_result[0], span_result[1], label_map)

            label_map = {0:'Chemical', 1:'Disease', 2:'Other'}
            pred_entity, true_entity = utils.postprocess(entity_result[0], entity_result[1], label_map)

            batch_span_acc = performance.performance_acc(pred_span, true_span, logger)
            running_span_acc += (batch_span_acc - running_span_acc) / (batch_index + 1)

            batch_entity_acc = performance.performance_acc(pred_entity, true_entity, logger)
            running_entity_acc += (batch_entity_acc - running_entity_acc) / (batch_index + 1)

            progress_bar_valid.update(1)
            progress_bar_valid.set_description("running_span_acc:{:.2f}, running_entity_acc:{:.2f} : epoch:{}".
                                        format(running_span_acc, running_entity_acc, epoch))


        print("running_span_acc: {:.2f}".format(running_span_acc))
        print("running_entity_acc: {:.2f}".format(running_entity_acc))

        running_acc = running_span_acc + running_entity_acc

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
    train(parameters, "autoner")

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))
        

if __name__ == '__main__':                                                                                                                        
    main()


