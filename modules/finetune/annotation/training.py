import pandas as pd
import spacy
import scispacy
import os
import time
import pickle
import numpy as np
from collections import defaultdict
import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import get_scheduler
from accelerate import Accelerator
import evaluate


from tqdm.auto import tqdm

from utils import utils

from dataloader import Dataloader
from model2 import Model

from measure import performance
import json
import spacy
import pandas

import pdb

pd.set_option('display.max_colwidth', None)

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("sentencizer")


def sentence_split(doc, offset=False):
    doc = nlp(doc)
    if offset == False:
        sents = [sent.text for sent in doc.sents]
    else:
        sents = [(sent.text, sent.start_char) for sent in doc.sents]

    return sents


def tokenize(text, offset=False):
    doc = nlp(text)
    if offset == False:
        tokens = [token.text for token in doc if token.text != '\n']
    else:
        tokens = [(token.text, token.idx)
                  for token in doc if token.text != '\n']
    return tokens


def train(df_train_pos, df_train_neg, parameters, model_name_suffix):

    # logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter(
        "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

    model_dir = parameters['model_dir']

    handler2 = logging.FileHandler(
        filename=os.path.join(model_dir, "test.log"))
    handler2.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(handler1)
    logger.addHandler(handler2)

    dataloader = Dataloader(df_train_pos, df_train_neg, parameters)

    train_dataloader, valid_dataloader, test_dataloader = dataloader.load_data()

    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    pu_model = Model(parameters, logger)

    if parameters['restore_model'] == True:
        model_path = parameters['restore_model_path']
        pu_model.load_state_dict(torch.load(
            model_path, map_location=torch.device(device)))

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
        logger.debug("running_acc: {:.2f}".format(running_acc))

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

        torch.save(unwrapped_model.state_dict(), os.path.join(
            OUTPUT_PATH, f"model_last_{model_name_suffix}.pth"))

        if best_update:
            torch.save(unwrapped_model.state_dict(), os.path.join(
                OUTPUT_PATH, f"model_best_{model_name_suffix}.pth"))

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


def setup_finetune_dataset(parameters):

    negative_sample_ratio = 10
    max_entity_token_len = 5
    json_path = parameters["es_dump_path"]

    with open(json_path) as fp:
        json_data = json.load(fp)

    entity_names = parameters["entity_names"]
    etype2label = {name: k for k, name in enumerate(entity_names)}
    label2etype = {k: name for k, name in enumerate(entity_names)}

    ids = json_data["pmid"].keys()

    data = defaultdict(list)

    for id_ in sorted(ids):
        pmid = json_data["pmid"][id_]
        text = json_data["text"][id_]
        entities = json_data["entities"][id_]
        sents = sentence_split(text.strip(), offset=True)

        for k, (sent, offset) in enumerate(sents):
            positive_sample_count = 0
            negative_sample_count = 0

            mentions = []
            for entity in entities:
                if entity["end_char"] < offset or entity["start_char"] >= offset + len(sent):
                    continue
                mention = entity["mention"]
                etype = entity["entityType"]
                cui = entity["cui"]
                start_char = entity["start_char"]
                end_char = entity["end_char"]

                start_char -= offset
                end_char -= offset

                mentions.append(mention)
                assert (mention == sent[start_char:end_char])

                data["mention"].append(mention)
                data["start_char"].append(start_char)
                data["end_char"].append(end_char)
                data["text"].append(sent)
                data["label"].append(etype2label[etype])
                data["cui"].append(cui)
                data["pmid"].append(f"{pmid}_{k}")
                positive_sample_count += 1

            # sample negative tokens

            tokens = tokenize(sent, offset=True)
            n = len(tokens)
            for i in range(positive_sample_count * negative_sample_ratio):
                token_start = np.random.randint(0, n)
                token_len = np.random.randint(0, max_entity_token_len + 1)
                token_end = min(token_start + token_len, n - 1)

                start_char = tokens[token_start][1]
                end_char = tokens[token_end][1] + len(tokens[token_end][0])

                assert (0 <= start_char)
                assert (end_char <= len(sent))
                assert (start_char < end_char)
                mention = sent[start_char:end_char]

                # print(f"{start_char}:{end_char}:{len(sent)}:{mention}")

                if not mention in mentions:
                    data["mention"].append(mention)
                    data["start_char"].append(start_char)
                    data["end_char"].append(end_char)
                    data["text"].append(sent)
                    data["label"].append(len(etype2label))
                    data["cui"].append("NA")
                    data["pmid"].append(f"{pmid}_{k}")
                    negative_sample_count += 1

    df_train = pandas.DataFrame(data)

    # shuffle training dataset
    df_train = df_train.sample(frac=1)

    return df_train


def finetune_classifier(parameters):

    # add negative label class
    parameters["positive_class_num"] += 1
    df_train = setup_finetune_dataset(parameters)

    print("retrain_classifier with unknown samples")
    train(df_train, None, parameters, model_name_suffix="final")

    model_dir = parameters["model_dir"]

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

    # By using the positive and the negative samples extractd in step 1, we train a NN classifier
    # to classify each unknown sample whether it is one of the possitive classes or negative class
    finetune_classifier(parameters)

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()
