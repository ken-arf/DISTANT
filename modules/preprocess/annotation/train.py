
import sys
import os
import itertools
import logging
from io import StringIO
import pickle

import numpy as np
import math
import time
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
from accelerate import Accelerator

from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
from datasets.features.features import Sequence
from datasets.features.features import ClassLabel

import pdb


class PUTrain:

    def __init__(self, df_train_pos, df_train_neg, parameters):


        self.df_train_pos = df_train_pos
        self.df_train_neg = df_train_neg

        self.parameters = parameters
        self.myseed = self.parameters["seed"] 
        model_checkpoint = self.parameters["model_checkpoint"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.load_data()

    def load_data(self):


        # shuffle dataset
        df_train_pos = self.df_train_pos.sample(frac=1, replace=False, random_state=self.myseed)

        n = len(self.df_train_pos)
        val_size = int(n * 0.1)

        df_train_dataset_ = df_train_pos.iloc[:n-val_size]
        df_valid_dataset_ = df_train_pos.iloc[n-val_size:]

        train_dataset = Dataset.from_pandas(df_train_dataset_)
        valid_dataset = Dataset.from_pandas(df_valid_dataset_)
        test_dataset = Dataset.from_pandas(self.df_train_neg)

        self.datasets = DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset
            })

        
        tokenized_datasets = self.datasets.map(
                self.tokenized_and_align_labels,
                batched = True,
                remove_columns = self.datasets["train"].column_names
                )


        train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size = 4,
        )

        valid_dataloader = DataLoader(
            tokenized_datasets["valid"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size = 4,
        )

        pdb.set_trace()
        for batch in train_dataloader:
            print(batch)
            break

    def data_collator(self, features):

        batch = self.tokenizer.pad(
            features,
            padding = True,
            #max_length=max_length,
            pad_to_multiple_of = None,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors= None,
        )

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch

    def tokenized_and_align_labels(self, examples):


        tokenized_inputs = self.tokenizer(
                    examples["text"],
                    truncation = True,
                    max_length = 512,
                    is_split_into_words = False,
                    #return_tensors = "pt",
                    return_offsets_mapping = True,
                    return_overflowing_tokens = True,
        )


        print(f"The examples gave {len(tokenized_inputs['input_ids'])} features.")
        print(f"Here is where each comes from: {tokenized_inputs['overflow_to_sample_mapping']}.")

        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")

        offset_mapping = tokenized_inputs.pop("offset_mapping")

        print(tokenized_inputs.keys())


        start_positions = []
        end_positions = []
        labels = []
        
        for index, (sample_id, offset) in enumerate(zip(sample_map, offset_mapping)):
            
            start_char = int(examples["start_chars"][sample_id])
            end_char = int(examples["start_chars"][sample_id])
            label = examples["label"][sample_id]


            sequence_ids = tokenized_inputs.sequence_ids(index)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 0:
                idx += 1
            context_start = idx

            while sequence_ids[idx] == 0:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] <= start_char and offset[context_end][1] >= end_char:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_position = idx - 1

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_position = idx + 1
            
            else:
                if offset[context_start][0] <= start_char:
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_position = idx - 1
                    end_position = context_end + 1

                elif offset[conttext_end][1] >= end_char:

                    start_position = contex_start
                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_position = idx + 1


            start_positions.append(start_position)
            end_positions.append(end_position)
            labels.append(label)

    
        tokenized_inputs["start_positions"] = start_positions
        tokenized_inputs["end_positions"] = end_positions
        tokenized_inputs["labels"] = labels

        return tokenized_inputs


    def train(self):
        pass
