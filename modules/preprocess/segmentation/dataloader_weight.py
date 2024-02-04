
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

    def __init__(self, data, params):

        self.data = data

        self.params = params
        self.myseed = self.params["seed"]

        model_checkpoint = self.params["model_checkpoint"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        if torch.cuda.is_available() and params['gpu'] >= 0:
            self.device = "cuda"
        else:
            self.device = "cpu"
       
        # Check that MPS is available
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                      "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                      "and/or you do not have an MPS-enabled device on this machine.")
        else:
            self.device = torch.device("mps")


        self.device = torch.device("cpu")


    def load_data_train_only(self):


        texts = self.data['text']
        labels = self.data['label']

        train_dataset = {'text': texts, 'label': labels}

        train_dataset = Dataset.from_dict(train_dataset)

        self.datasets = DatasetDict({
            "train": train_dataset,
        })

        tokenized_datasets = self.datasets.map(
            self.tokenized_and_align_labels,
            batched=True,
            remove_columns=self.datasets["train"].column_names
        )

        train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.params["train_batch_size"],
        )

        return train_dataloader

    def load_data_preset(self):

        train_dataset = Dataset.from_dict(self.data['train'])
        valid_dataset = Dataset.from_dict(self.data['val'])

        self.datasets = DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
        })

        tokenized_datasets = self.datasets.map(
            self.tokenized_and_align_labels,
            batched=True,
            remove_columns=self.datasets["train"].column_names
        )

        train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.params["train_batch_size"],
        )

        valid_dataloader = DataLoader(
            tokenized_datasets["valid"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.params["valid_batch_size"],
        )

        return train_dataloader, valid_dataloader


    def load_data(self):

        val_size_ratio = 0.1

        # shuffle dataset
        random.seed(self.myseed)

        texts = self.data['text']
        labels = self.data['label']
        index = list(range(len(texts)))
        random.shuffle(index)

        stexts = [texts[i] for i in index]
        slabels = [labels[i] for i in index]

        n = len(texts)
        val_size = int(n * val_size_ratio)
        train_size = n - val_size

        train_stexts = stexts[:train_size]
        valid_stexts = stexts[train_size:]

        train_slabels = slabels[:train_size]
        valid_slabels = slabels[train_size:]

        train_dataset = {'text': train_stexts, 'label': train_slabels}
        valid_dataset = {'text': valid_stexts, 'label': valid_slabels}

        train_dataset = Dataset.from_dict(train_dataset)
        valid_dataset = Dataset.from_dict(valid_dataset)

        self.datasets = DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
        })

        tokenized_datasets = self.datasets.map(
            self.tokenized_and_align_labels,
            batched=True,
            remove_columns=self.datasets["train"].column_names
        )

        train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.params["train_batch_size"],
        )

        valid_dataloader = DataLoader(
            tokenized_datasets["valid"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.params["valid_batch_size"],
        )

        return train_dataloader, valid_dataloader

    def data_collator(self, features):


        batch = self.tokenizer.pad(
            features,
            padding=True,
            # max_length=max_length,
            pad_to_multiple_of=None,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors=None,
        )

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]

        label_pad_token_id = -100
        label_name = "label" if "label" in features[0].keys() else "labels"

        labels = [feature[label_name]
                  for feature in features] if label_name in features[0].keys() else None

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

        # collate weights
        weights = [feature['weights'] for feature in features]

        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch['weights'] = [
                list(weight) + [0] * (sequence_length - len(weight)) for weight in weights
            ]
        else:
            batch['weights'] = [
                [0] * (sequence_length - len(weight)) + list(weight) for weight in weights
            ]


        #print([len(l) for l in batch['labels']])
        #print([len(l) for l in batch['weights']])

        batch = {k: torch.tensor(v, dtype=torch.int64).to(self.device)
                 for k, v in batch.items()}
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

        assert (len(word_ids) == len(new_labels))
        return new_labels

    def tokenized_and_align_labels(self, examples):

        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            is_split_into_words=True,
            # return_tensors = "pt",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )

        # print(f"The examples gave {len(tokenized_inputs['input_ids'])} features.")
        # print(f"Here is where each comes from: {tokenized_inputs['overflow_to_sample_mapping']}.")


        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_inputs.pop("offset_mapping")

        labels = []
        weights = []
        for index, (sample_id, offset) in enumerate(zip(sample_map, offset_mapping)):

            sample_label = examples["label"][sample_id]
            sample_weight = examples["weight"][sample_id]
            word_ids = tokenized_inputs.word_ids(index)

            aligned_label = self._align_labels_with_tokens(
                sample_label, word_ids)
            labels.append(aligned_label)

            aligned_weight = self._align_labels_with_tokens(
                sample_weight, word_ids)
            weights.append(aligned_weight)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["weights"] = weights

        return tokenized_inputs
