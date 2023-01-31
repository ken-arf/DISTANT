
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
from torch import nn, Tensor
import torch.nn.functional as F

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


class PU_Model(nn.Module):

    def __init__(self, dataloader, params, logger):
        super(PU_Model, self).__init__()

        self.dataloader = dataloader
        self.params = params
        self.logger = logger
        self.myseed = self.params["seed"] 
        model_checkpoint = self.params["model_checkpoint"]

        if torch.cuda.is_available() and self.params['gpu'] >= 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # load bare BertModel
        self.bert_model = BertModel.from_pretrained(model_checkpoint).to(self.device)
        
        self.dropout = nn.Dropout(self.params['dropout_rate'])

        hidden_size = 100
        #self.tanh = nn.Tanh()
        #self.linear = nn.Linear(self.params['embedding_dim'], hidden_size).to(self.device)
        #self.linear = nn.Linear(hidden_size, self.params['class_num']).to(self.device)

        self.linear = nn.Sequential(
          nn.Linear(self.params['embedding_dim'] * 3, hidden_size),
          nn.Tanh(),
          nn.Linear(hidden_size, self.params['class_num'])
        ).to(self.device)

        # cross entropy loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, **kargs):

        input_ids = kargs["input_ids"]
        token_type_ids = kargs["token_type_ids"]
        attention_mask = kargs["attention_mask"]
        start_pos = kargs["start_positions"]
        end_pos = kargs["end_positions"]
        labels = kargs["labels"]

        #Extract outputs from the body
        bert_outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        #Add custom layers
        last_hidden_output = bert_outputs['last_hidden_state']
        bert_sequence_output = self.dropout(last_hidden_output) #outputs[0]=last hidden state


        features = []
        for k, (start, end) in enumerate(zip(start_pos, end_pos)):
            feature = bert_sequence_output[k, start:end+1, :]
            if feature.shape[0] == 1:
                head_feature = feature[0,:]
                mean_feature = feature[0,:]
                tail_feature = feature[0,:]
            elif feature.shape[0] == 2:
                head_feature = feature[0,:]
                mean_feature = torch.mean(feature, 0)
                tail_feature = feature[-1,:]
            else:
                head_feature = feature[0,:]
                mean_feature = torch.mean(feature[1:-1], 0)
                tail_feature = feature[-1,:]

            features.append(torch.cat((head_feature, mean_feature, tail_feature)))

        features_pt = torch.stack(features)

        # logit
        logit = self.linear(features_pt)
        loss = self.loss(logit, labels)

        return loss


    def decode(self, **kargs):

        input_ids = kargs["input_ids"]
        token_type_ids = kargs["token_type_ids"]
        attention_mask = kargs["attention_mask"]
        start_pos = kargs["start_positions"]
        end_pos = kargs["end_positions"]
        labels = kargs["labels"]

        #Extract outputs from the body
        bert_outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        #Add custom layers
        last_hidden_output = bert_outputs['last_hidden_state']
        bert_sequence_output = self.dropout(last_hidden_output) #outputs[0]=last hidden state


        features = []
        for k, (start, end) in enumerate(zip(start_pos, end_pos)):
            feature = bert_sequence_output[k, start:end+1, :]
            if feature.shape[0] == 1:
                head_feature = feature[0,:]
                mean_feature = feature[0,:]
                tail_feature = feature[0,:]
            elif feature.shape[0] == 2:
                head_feature = feature[0,:]
                mean_feature = torch.mean(feature, 0)
                tail_feature = feature[-1,:]
            else:
                head_feature = feature[0,:]
                mean_feature = torch.mean(feature[1:-1], 0)
                tail_feature = feature[-1,:]

            features.append(torch.cat((head_feature, mean_feature, tail_feature)))

        features_pt = torch.stack(features)

        # logit
        logit = self.linear(features_pt)
        predicts = torch.argmax(logit, axis=1)

        # probability
        probs = F.softmax(logit, dim=1)

        return predicts.cpu().detach().numpy(), probs.cpu().detach().numpy()

