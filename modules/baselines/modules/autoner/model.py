
import sys
import os
import itertools
import logging
import pickle

import numpy as np
import math
import time
import gensim


from dataclasses import dataclass
from sklearn.metrics import accuracy_score

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


from torch.optim import AdamW

from tqdm.auto import tqdm

from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
from datasets.features.features import Sequence
from datasets.features.features import ClassLabel

import pdb


class AutoNER(nn.Module):

    def __init__(self, params, logger):
        super(AutoNER, self).__init__()

        self.params = params
        self.logger = logger
        self.myseed = self.params["seed"] 

        if torch.cuda.is_available() and self.params['gpu'] >= 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load word2vec pre-train model
        word2vec_model = self.params["word2vec"]
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
        #model = gensim.models.Word2Vec.load('./word2vec_pretrain_v300.model')

        # embedding 
        self.word_embed_size = 200
        weights = torch.FloatTensor(self.w2v_model.vectors)
        self.word_embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=-100)
        self.word_embedding.requires_grad = False

        self.char_embed_size = 20 
        self.char_embedding = nn.Embedding(256, self.char_embed_size, padding_idx=-100)

        # lstm
        self.char_lstm_input_size = self.char_embed_size
        self.char_lstm_hidden_size = 20
        self.char_lstm_num_layers = 2
        self.char_bilstm = nn.LSTM(self.char_lstm_input_size, 
                                    self.char_lstm_hidden_size,
                                    self.char_lstm_num_layers,
                                    batch_first=True,
                                    bidirectional=True)

        self.word_lstm_input_size = self.word_embed_size + self.char_lstm_hidden_size * 4
        self.word_lstm_hidden_size = 50 
        self.word_lstm_num_layers = 4
        self.word_bilstm = nn.LSTM(self.word_lstm_input_size, 
                                    self.word_lstm_hidden_size,
                                    self.word_lstm_num_layers,
                                    batch_first=True,
                                    bidirectional=True)

        # load bare BertModel
        self.dropout = nn.Dropout(self.params['dropout_rate'])
        hidden_size = self.params['hidden_size']

        self.span_linear = nn.Sequential(
          nn.Dropout(self.params["dropout_rate"]),
          nn.Linear(self.word_lstm_hidden_size * 2, hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size, 2)
        ).to(self.device)

        self.entity_linear = nn.Sequential(
          nn.Dropout(self.params["dropout_rate"]),
          nn.Linear(self.word_lstm_hidden_size *2*3, hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size, self.params["class_num"])
        ).to(self.device)

        # cross entropy loss
        self.span_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.entity_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def _process_char_lstm(self, input_char_ids, input_char_lengths):


        result = []
        # batch_len, word seuqnce_len, char sequence_len
        bs, wseq_len, cseq_len = input_char_ids.shape
        for i in range(bs):
            input_char_id = input_char_ids[i]
            input_char_len = input_char_lengths[i]

            word_vec = []
            for j in range(wseq_len):
                input_char = input_char_id[j]
                input_len = input_char_len[j]

                if input_len != -100:
                    input_char = input_char[:input_len]
                    input_embed = self.char_embedding(input_char)
                    output, _ = self.char_bilstm(input_embed)
                    out = torch.concat([output[0], output[-1]], dim=0)
                    word_vec.append(out)
                else:
                    word_vec.append(torch.zeros(self.char_lstm_hidden_size * 4).to(self.device))
            result.append(torch.stack(word_vec, dim=0))
                    
        result = torch.stack(result, dim=0)
        return result

    def _extract_span(self, spans):

        runs = []
        run_start = spans[0]
        for i in range(len(spans)-1):
            if spans[i]+1 != spans[i+1]:
                # store
                runs.append([run_start, spans[i]+1])
                run_start = spans[i+1]
        runs.append([run_start, spans[-1]+1])

        return runs
        
    def _comp_entity_loss(self, features, span, ent_labels):

        features = features[span[0]:span[1]]
        head = features[0]
        mean = torch.mean(features, dim=0)
        tail = features[-1]
        cat_features = torch.cat([head, mean, tail])
        logit = self.entity_linear(cat_features)
        predict = torch.argsort(logit, descending=True, dim=-1)

        if ent_labels == None:
            return predict.tolist()

        if 2 in ent_labels and len(ent_labels) == 1:
            pass
        else:
            # if others(2) is included in true labels with other positve labels (chemicals:0, disease:1)
            # others(2) should be remove to prevent injecting noisy single.
            ent_labels.remove(2)

        true_y = torch.zeros(3)
        true_y.index_fill_(0, torch.tensor(ent_labels), 1)
        true_y = F.normalize(true_y, dim=0).to(self.device)
        pred_y = predict[:len(ent_labels)].tolist()
        pred_y = sorted(pred_y)

        loss = self.entity_loss(logit, true_y)

        return loss, (pred_y, ent_labels)

    def forward(self, **kargs):

        input_ids = kargs["input_ids"]
        input_char_ids = kargs["input_char_ids"]
        input_char_lengths = kargs["input_char_lengths"]
        slength = kargs["slength"]
        span_label = kargs["span"]

        bio_label = {}
        for k in range(3):
            bio_label[k] = kargs[f"bio_{k}"]

        char_embed = self._process_char_lstm(input_char_ids, input_char_lengths)

        input_ids[input_ids==-100] = 0
        input_embed = self.word_embedding(input_ids)
        padded_input = torch.cat((input_embed, char_embed), dim=-1)

        # lstm
        packed_input = pack_padded_sequence(padded_input, slength.cpu().detach(), batch_first=True)
        packed_output, _  = self.word_bilstm(packed_input)
        padded_output, length = pad_packed_sequence(packed_output, batch_first=True)

        span_output = self.span_linear(padded_output)
    
        span_loss = self.span_loss(torch.transpose(span_output,1,2), span_label)

        ent_loss = None
        bs, slen = span_label.shape
        for i in range(bs):

            features = padded_output[i]
            span_index = torch.where(span_label[i] == 1)
            if torch.numel(span_index[0]) == 0:
                continue
            spans = self._extract_span(span_index[0])

            for span in spans:
                ent_labels = []
                for k in range(3):
                    bio = bio_label[k][i]
                    match = torch.all(torch.tensor([k]*(span[1]-span[0])).to(self.device)==bio[span[0]:span[1]])
                    if match.item():
                        ent_labels.append(k)

                if len(ent_labels) == 0:
                    continue

                if ent_loss == None:
                    loss, _ = self._comp_entity_loss(features, span, ent_labels)
                    ent_loss = loss
                else:
                    loss, _ = self._comp_entity_loss(features, span, ent_labels)
                    ent_loss += loss
                    
        return span_loss, ent_loss 


    def decode(self, **kargs):

        input_ids = kargs["input_ids"]
        input_char_ids = kargs["input_char_ids"]
        input_char_lengths = kargs["input_char_lengths"]
        slength = kargs["slength"]
        span_label = kargs["span"]

        bio_label = {}
        for k in range(3):
            bio_label[k] = kargs[f"bio_{k}"]

        char_embed = self._process_char_lstm(input_char_ids, input_char_lengths)

        input_ids[input_ids==-100] = 0
        input_embed = self.word_embedding(input_ids)
        padded_input = torch.cat((input_embed, char_embed), dim=-1)

        # lstm
        packed_input = pack_padded_sequence(padded_input, slength.cpu().detach(), batch_first=True)
        packed_output, _  = self.word_bilstm(packed_input)
        padded_output, length = pad_packed_sequence(packed_output, batch_first=True)


        span_output = self.span_linear(padded_output)
        predict = torch.argmax(span_output, dim=-1)
        mask = span_label != -100

        true_span = span_label[mask]
        pred_span = predict[mask]

        pred_ys = []
        true_ys = []
        bs, slen = span_label.shape
        for i in range(bs):

            features = padded_output[i]
            span_index = torch.where(span_label[i] == 1)
            if torch.numel(span_index[0]) == 0:
                continue
            spans = self._extract_span(span_index[0])

            for span in spans:
                ent_labels = []
                for k in range(3):
                    bio = bio_label[k][i]
                    match = torch.all(torch.tensor([k]*(span[1]-span[0])).to(self.device)==bio[span[0]:span[1]])
                    if match.item():
                        ent_labels.append(k)

                if len(ent_labels) == 0:
                    continue
                _, (pred_y, true_y) = self._comp_entity_loss(features, span, ent_labels)
                pred_ys += pred_y
                true_ys += true_y


        return (pred_span[None,:], true_span[None,:]), (torch.LongTensor([pred_ys]), torch.LongTensor([true_ys]))

    def predict(self, **kargs):


        input_ids = kargs["input_ids"]
        input_char_ids = kargs["input_char_ids"]
        input_char_lengths = kargs["input_char_lengths"]
        slength = kargs["slength"]

        #bio_label = {}
        #for k in range(3):
        #    bio_label[k] = kargs[f"bio_{k}"]

        char_embed = self._process_char_lstm(input_char_ids, input_char_lengths)

        input_ids[input_ids==-100] = 0
        input_embed = self.word_embedding(input_ids)
        padded_input = torch.cat((input_embed, char_embed), dim=-1)

        # lstm
        packed_input = pack_padded_sequence(padded_input, slength.cpu().detach(), batch_first=True)
        packed_output, _  = self.word_bilstm(packed_input)
        padded_output, length = pad_packed_sequence(packed_output, batch_first=True)

        span_output = self.span_linear(padded_output)
        span_pred = torch.argmax(span_output, dim=-1)

        pred_spans = []
        pred_ys = []
        bs, slen = span_pred.shape
        for i in range(bs):

            features = padded_output[i]
            span_index = torch.where(span_pred[i] == 1)
            if torch.numel(span_index[0]) == 0:
                continue
            spans = self._extract_span(span_index[0])

            for span in spans:
                pred_y = self._comp_entity_loss(features, span, None)
                if pred_y[0] != 2:
                    pred_spans.append((span[0].item(), span[1].item()))
                    pred_ys.append(pred_y[0])
            
        return pred_spans, pred_ys
