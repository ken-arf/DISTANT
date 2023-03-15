

import sys
import os
import itertools
import logging
from io import StringIO
import pickle

import pandas as pd
import numpy as np
import math
import time
import random
import json
import evaluate

import tempfile

from dataclasses import dataclass
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from tqdm.auto import tqdm

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import get_scheduler
from transformers import default_data_collator
from transformers import BertModel
from accelerate import Accelerator
import gensim
import torch
import torch.nn as nn

from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
from datasets.features.features import Sequence
from datasets.features.features import ClassLabel

import pdb


class Dataloader:

    def __init__(self, data, params, logger):


        self.data = data

        self.params = params
        self.logger = logger
        self.myseed = self.params["seed"] 

        self.span2int= {'O': 0, 'I': 1}
        self.label2int = {}

        for key, v in self.params['ent2int'].items():
            self.label2int[key.upper()] = v
        self.label2int['O'] = -100


        # Load word2vec pre-train model
        word2vec_model = self.params["word2vec"]
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
        #model = gensim.models.Word2Vec.load('./word2vec_pretrain_v300.model')


    def load_data(self):

        # shuffle dataset
        random.seed(self.myseed)

        texts  = self.data['text']

        bio = {}
        for i in range(self.params['class_num']):
            bio[i] = [[self.label2int[label] for label in labels] for labels in self.data[f'bio_labels_{i}']] 

        spans = [[self.span2int[label] for label in labels] for labels in self.data['spans']] 

        index = list(range(len(texts)))
        random.shuffle(index)

        texts = [texts[i] for i in index]
        for k in range(self.params['class_num']):
            bio[k] = [bio[k][i] for i in index]
        spans = [spans[i] for i in index]

        n = len(texts)
        val_size = int(n * 0.1)
        train_size = n - val_size

        train_texts = texts[:train_size]
        valid_texts = texts[train_size:]

        train_bio = {}
        valid_bio = {}
        for k in range(self.params['class_num']):
            train_bio[k] = bio[k][:train_size]
            valid_bio[k] = bio[k][train_size:]
    
        train_spans = spans[:train_size]
        valid_spans = spans[train_size:]


        train_dataset = {'text': train_texts, 'span': train_spans}
        train_bio = {f'bio_{k}': train_bio[k] for k in range(self.params['class_num'])}
        train_dataset.update(train_bio)

        valid_dataset = {'text': valid_texts, 'span': valid_spans}
        valid_bio = {f'bio_{k}': valid_bio[k] for k in range(self.params['class_num'])}
        valid_dataset.update(valid_bio)


        train_dataset = Dataset.from_dict(train_dataset)
        valid_dataset = Dataset.from_dict(valid_dataset)

        self.datasets = DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
            })


        tokenized_datasets = self.datasets.map(
                self.tokenized_and_align_labels,
                batched = True,
                remove_columns = ["text"],
                )


        train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size = self.params["train_batch_size"],
        )


        valid_dataloader = DataLoader(
            tokenized_datasets["valid"],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size = self.params["valid_batch_size"],
        )

        return train_dataloader, valid_dataloader


    def data_collator(self, features):

        # batch size
        bs = len(features)

        length = []
        for i in range(bs):
            l = len(features[i]['input_ids'])
            length.append(l)

        features = features.copy()

        # sort sample by sentence length in decending order
        sindex = np.argsort(length)
        sindex = sindex[::-1]
        slength = [length[i] for i in sindex]

        input_ids = [features[i]['input_ids'] for i in sindex]
        input_char_ids = [features[i]['input_char_ids'] for i in sindex]
        spans = [features[i]['span'] for i in sindex]
        bio_labels = {}
        for k in range(3):
            bio_labels[k] = [features[i][f'bio_{k}'] for i in sindex]
        
        # take maximum sequence
        sequence_length = slength[0]

        batch = {}
        batch['slength'] = slength
        batch['input_ids'] = [
            list(input_id) + [-100] * (sequence_length - len(input_id)) for input_id in input_ids
        ]
        batch['span'] = [
            list(span) + [-100] * (sequence_length - len(span)) for span in spans
        ]
        for k in range(3):
            batch[f'bio_{k}'] = [
                list(bio) + [-100] * (sequence_length - len(bio)) for bio in bio_labels[k]
            ]

        # compute maximum char-length within the mini-batch
        max_char_len = 0
        for input_char_id in input_char_ids:
            for char_id in input_char_id:
                char_len = len(char_id)
                if char_len > max_char_len:
                    max_char_len = char_len 
    

        batch['input_char_ids'] = []
        for input_char_id in input_char_ids:
            padded_char_id = [
                char_id + [-100] * (max_char_len - len(char_id)) for char_id in input_char_id
            ]

            padded_char_id += [
                [-100] * max_char_len for i in range(sequence_length - len(input_char_id))
            ]
            batch['input_char_ids'].append(padded_char_id)

        batch['input_char_lengths'] = []
        for input_char_id in input_char_ids:
            char_len = [ len(char_id)  for char_id in input_char_id ]
            char_len += [ -100 for i in range(sequence_length - len(input_char_id))]
            batch['input_char_lengths'].append(char_len)

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


    def tokenized_and_align_labels(self, examples):

        tokenized_input = {}

        input_ids = []
        UNK_id = self.w2v_model.get_index('unk')
        for text in examples["text"]:
            ids = [self.w2v_model.get_index(word.lower()) if word.lower() in self.w2v_model else UNK_id for word in text]
            input_ids.append(ids)

        # debug
        for input_txt, input_id in zip(examples["text"], input_ids):
            #print(np.sum(np.array(input_id) == UNK_id))
            self.logger.info(np.sum(np.array(input_id) == UNK_id))
            if np.sum(np.array(input_id) == UNK_id) > 0:
                args = np.where(np.array(input_id)==UNK_id)[0].tolist()
                #print([input_txt[arg] for arg in args])
                self.logger.info([input_txt[arg] for arg in args])


        input_char_ids = []
        for text in examples["text"]:
            words = []
            for word in text:
                char_ids = [ord(ch) for ch in list(word.lower())]
                words.append(char_ids)
            input_char_ids.append(words)

        tokenized_input["input_ids"] = input_ids
        tokenized_input["input_char_ids"] = input_char_ids
        tokenized_input["span"] = examples["span"].copy()
        for i in range(3):
            tokenized_input[f"bio_{i}"] = examples[f"bio_{i}"].copy()
            
        return tokenized_input

