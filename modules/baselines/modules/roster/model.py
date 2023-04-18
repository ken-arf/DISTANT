
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



class RoSTER(nn.Module):

    def __init__(self, params, logger):
        super(RoSTER, self).__init__()

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

        hidden_size = self.params['hidden_size']

        self.linear = nn.Sequential(
          nn.Linear(self.params['embedding_dim'], hidden_size),
          nn.Tanh(),
          nn.Linear(hidden_size, self.params['class_num'])
        ).to(self.device)

        # cross entropy loss
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def calc_loss(self, probs, labels, epoch):

        #pdb.set_trace()

        mask = labels != -100
        probs = probs[mask]
        labels = labels[mask]
        n = probs.shape[0]

        q = 0.7
        tau = 0.7

        loss = None
        for i in range(n):
            p = probs[i,labels[i]]
            #if epoch == 0:
            if epoch <= 2:
                w = 1.0
            else:
                w = 1.0 if p > tau else 0.0

            l = w * torch.div(1.0 - torch.pow(p, q), q)

            if i == 0:
                loss = l
            else:
                loss += l

        #loss = torch.div(loss, n)

        return loss 

    def forward(self, **kargs):

        input_ids = kargs["input_ids"]
        token_type_ids = kargs["token_type_ids"]
        attention_mask = kargs["attention_mask"]
        labels = kargs["labels"]
        epoch = kargs["epoch"]

        #Extract outputs from the body
        bert_outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        #Add custom layers
        last_hidden_output = bert_outputs['last_hidden_state']
        bert_sequence_output = self.dropout(last_hidden_output) #outputs[0]=last hidden state

        # logit
        logit = self.linear(bert_sequence_output)
        # probability
        probs = F.softmax(logit, dim=2)

        loss = self.calc_loss(probs, labels, epoch)

        #loss = self.loss(torch.transpose(logit,1,2), labels)

        return probs.cpu().detach().numpy(), loss


    def decode(self, **kargs):

        input_ids = kargs["input_ids"]
        token_type_ids = kargs["token_type_ids"]
        attention_mask = kargs["attention_mask"]
        labels = kargs["labels"]

        #Extract outputs from the body
        bert_outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        #Add custom layers
        last_hidden_output = bert_outputs['last_hidden_state']
        bert_sequence_output = self.dropout(last_hidden_output) #outputs[0]=last hidden state


        # logit
        logit = self.linear(bert_sequence_output)
        loss = self.loss(torch.transpose(logit,1,2), labels)

        # probability
        probs = F.softmax(logit, dim=2)
    
        # prediction
        predicts = torch.argmax(logit, dim=2)

        return predicts.cpu().detach().numpy(), probs.cpu().detach().numpy()

    def predict(self, **kargs):

        input_ids = kargs["input_ids"]
        token_type_ids = kargs["token_type_ids"]
        attention_mask = kargs["attention_mask"]

        #Extract outputs from the body
        bert_outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        #Add custom layers
        last_hidden_output = bert_outputs['last_hidden_state']
        bert_sequence_output = self.dropout(last_hidden_output) #outputs[0]=last hidden state


        # logit
        logit = self.linear(bert_sequence_output)

        # probability
        probs = F.softmax(logit, dim=2)
    
        # prediction
        predicts = torch.argmax(logit, dim=2)

        return predicts.cpu().detach().numpy(), probs.cpu().detach().numpy()


class RoSTER_ENS(RoSTER):

    def __init__(self, params, logger):
        super(RoSTER_ENS, self).__init__(params, logger)
        # cross entropy loss
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")


    def forward(self, **kargs):

        #pdb.set_trace()

        input_ids = kargs["input_ids"]
        token_type_ids = kargs["token_type_ids"]
        attention_mask = kargs["attention_mask"]
        labels = kargs["labels"]
        softlabels = kargs["softlabels"]

        # conver numpy array to tensor
        softlabels = torch.tensor(softlabels).to(self.device)

        #Extract outputs from the body
        bert_outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        #Add custom layers
        last_hidden_output = bert_outputs['last_hidden_state']
        bert_sequence_output = self.dropout(last_hidden_output) #outputs[0]=last hidden state

        # logit
        logit = self.linear(bert_sequence_output)
        # probability
        probs = F.log_softmax(logit, dim=2)

        #loss = self.loss(torch.transpose(logit,1,2), labels)
        loss = self.kl_loss(probs, softlabels)

        return probs.cpu().detach().numpy(), loss

class RoSTER_self(RoSTER):

    def __init__(self, params, logger):
        super(RoSTER_self, self).__init__(params, logger)
        # cross entropy loss
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")


    def forward(self, **kargs):


        input_ids = kargs["input_ids"]
        aux_input_ids = kargs["aux_input_ids"]
        token_type_ids = kargs["token_type_ids"]
        attention_mask = kargs["attention_mask"]
        labels = kargs["labels"]
        softlabels = kargs["softlabels"]

        # conver numpy array to tensor
        softlabels = torch.tensor(softlabels).to(self.device)

        #Extract outputs from the body
        if kargs["use_aux_input"]:
            bert_outputs = self.bert_model(input_ids=aux_input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            bert_outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        #Add custom layers
        last_hidden_output = bert_outputs['last_hidden_state']
        bert_sequence_output = self.dropout(last_hidden_output) #outputs[0]=last hidden state

        # logit
        logit = self.linear(bert_sequence_output)
        # probability
        probs = F.log_softmax(logit, dim=2)

        #loss = self.loss(torch.transpose(logit,1,2), labels)
        loss = self.kl_loss(probs, softlabels)

        return probs.cpu().detach().numpy(), loss
