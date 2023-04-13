
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

from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
from datasets.features.features import Sequence
from datasets.features.features import ClassLabel

import pdb


class Dataloader:

    def __init__(self, dataset, params, logger):

        self.dataset = dataset
        self.params = params
        self.logger = logger

        self.myseed = self.params["seed"] 

        self.label2int = self.params["label2int"]

        model_checkpoint = self.params["model_checkpoint"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def load_data(self):

        # shuffle dataset
        random.seed(self.myseed)

        texts  = self.dataset['text']

        bio_label = [[self.label2int[label] for label in labels] for labels in self.dataset['bio_labels']] 

        index = list(range(len(texts)))
        random.shuffle(index)

        texts = [texts[i] for i in index]
        bio_label = [bio_label[i] for i in index]

        n = len(texts)
        val_size = int(n * 0.1)
        train_size = n - val_size

        train_texts = texts[:train_size]
        valid_texts = texts[train_size:]

        train_label = bio_label[:train_size]
        valid_label = bio_label[train_size:]

        train_dataset = {'text': train_texts}
        train_label = {'bio_label': train_label}
        train_dataset.update(train_label)

        valid_dataset = {'text': valid_texts}
        valid_label = {'bio_label': valid_label}
        valid_dataset.update(valid_label)


        train_dataset = Dataset.from_dict(train_dataset)
        valid_dataset = Dataset.from_dict(valid_dataset)

        self.datasets = DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
            })


        tokenized_datasets = self.datasets.map(
                self.tokenized_and_align_labels,
                batched = True,
                remove_columns = ["text", "bio_label"],
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


        batch = self.tokenizer.pad(
            features,
            padding = True,
            #max_length=max_length,
            pad_to_multiple_of = None,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors= None,
        )

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]

        label_pad_token_id = -100
        label_name = "label" if "label" in features[0].keys() else "labels"

        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        # collate labels
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]


        word_ids = [feature["word_ids"] for feature in features]
        # collate word_ids
        label_pad_token_id = -100
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["word_ids"] = [
                list(word_id) + [label_pad_token_id] * (sequence_length - len(word_id)) for word_id in word_ids
            ]
        else:
            batch["word_ids"] = [
                [label_pad_token_id] * (sequence_length - len(word_id)) + list(word_id) for word_id in word_ids
            ]


        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch

    def _align_labels_with_tokens(self, labels, word_ids):

        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                new_labels.append(label)

        assert(len(word_ids) == len(new_labels))
        return new_labels

    def tokenized_and_align_labels(self, examples):


        tokenized_inputs = self.tokenizer(
                    examples["text"],
                    truncation = True,
                    max_length = 512,
                    is_split_into_words = True,
                    #return_tensors = "pt",
                    return_offsets_mapping = True,
                    return_overflowing_tokens = True,
        )

        #print(f"The examples gave {len(tokenized_inputs['input_ids'])} features.")
        #print(f"Here is where each comes from: {tokenized_inputs['overflow_to_sample_mapping']}.")

        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_inputs.pop("offset_mapping")


        labels = []
        wordIds = []
        for index, (sample_id, offset) in enumerate(zip(sample_map, offset_mapping)):
            
            sample_label = examples["bio_label"][sample_id]
            word_ids = tokenized_inputs.word_ids(index)

            aligned_label = self._align_labels_with_tokens(sample_label, word_ids)
            labels.append(aligned_label)

            # change None to -100
            word_ids[0] = -100
            word_ids[-1] = -100
            wordIds.append(word_ids)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["word_ids"] = wordIds

        return tokenized_inputs

