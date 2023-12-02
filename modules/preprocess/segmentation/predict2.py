import pandas as pd
import spacy
import scispacy
import os
import time
import pickle
import numpy as np
from glob import glob
import logging

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import get_scheduler
from transformers import default_data_collator
from transformers import BertModel

from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
from datasets.features.features import Sequence
from datasets.features.features import ClassLabel

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer, LancasterStemmer


from tqdm.auto import tqdm

from utils import utils

from preprocess.segmentation.model2 import Model

from preprocess.segmentation.make_dataset import tokenize
from preprocess.segmentation.make_dataset import sentence_split

import dataclasses

import pdb


@dataclasses.dataclass
class Entity:
    text: str
    start: int = 0
    end: int = 0
    start_char: int = 0
    end_char: int = 0


class EntityExtraction:

    def __init__(self, config_yaml):

        # logging
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.ERROR)

        handler1 = logging.StreamHandler()
        handler1.setFormatter(logging.Formatter(
            "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

        self.logger.addHandler(handler1)

        # self.nlp = spacy.load("en_core_sci_lg")
        self.nlp = spacy.load("en_core_sci_sm")
        self.nlp.add_pipe("sentencizer")

        with open(config_yaml, 'r') as stream:
            params = utils._ordered_load(stream)
            self.params = params

        self.load_model()

    def load_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.params["model_checkpoint"])

        if torch.cuda.is_available() and self.params['gpu'] >= 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = Model(self.params, self.logger)

        model_dir = self.params["model_dir"]
        model_name = self.params["model_name"]
        restore_model_checkpoint = os.path.join(model_dir, model_name)

        self.model.load_state_dict(torch.load(
            restore_model_checkpoint, map_location=torch.device(self.device)))

    def get_entities(self, text):

        tokenized_input, aux_data = self.tokenize_text(text)
        prediction, prob = self.model.predict(**tokenized_input)

        entity_indexes = []
        span = []
        # search for 'B': 1
        n = len(prediction[0])

        pred = prediction[0]

        in_S = False
        in_BI = False
        for i, l in enumerate(pred):
            if l == 3:  # S
                if in_BI:
                    span_end = i - 1
                    entity_indexes.append([span_start, span_end])
                    in_BI = False
                if not in_S:
                    in_S = True
                    span_start = i
            elif l == 1:  # B
                if in_S:
                    span_end = i - 1
                    entity_indexes.append([span_start, span_end])
                    in_S = False
                in_BI = True
                span_start = i
            elif l == 2:  # I
                if not in_BI:
                    in_BI = True
                    span_start = i
            elif l == 0:  # O
                if in_BI:
                    span_end = i-1
                    entity_indexes.append([span_start, span_end])
                    in_BI = False
                elif in_S:
                    span_end = i-1
                    entity_indexes.append([span_start, span_end])
                    in_S = False

        if in_BI or in_S:
            span_end = i
            entity_indexes.append([span_start, span_end])

        # conver index to word_ids
        word_indexes = [[aux_data["word_ids"][index]
                         for index in span] for span in entity_indexes]

        entities = []
        for span_index in word_indexes:

            ##
            # work around when 1 is flaged at the first or the last token, which should never happen.
            if None in span_index:
                continue
            ##

            start = span_index[0]
            end = span_index[-1]+1

            tokens = aux_data["tokens"][start:end]
            offsets = aux_data["token_offsets"][start:end]
            start_char = offsets[0]
            end_char = offsets[-1] + len(tokens[-1])
            text = aux_data["text"][start_char:end_char]
            ent = Entity(text=text, start=start, end=end,
                         start_char=start_char, end_char=end_char)

            entities.append(ent)

        return entities

    def tokenize_text(self, text):

        tokens = tokenize(text, offset=True)

        token_text = [token[0] for token in tokens]
        token_offset = [token[1] for token in tokens]

        if len(tokens) > 512:
            print("Error, token length exceed 512")
            exit()

        tokenized_inputs = self.tokenizer(
            [token_text],
            truncation=True,
            max_length=512,
            is_split_into_words=True,
            # return_tensors = "pt",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )

        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_inputs.pop("offset_mapping")
        word_ids = tokenized_inputs.word_ids(0)

        tensor_input = {k: torch.tensor(v, dtype=torch.int64).to(
            self.device) for k, v in tokenized_inputs.items()}

        aux_data = {}
        aux_data["text"] = text
        aux_data["tokens"] = token_text
        aux_data["token_offsets"] = token_offset
        aux_data["offset_mapping"] = offset_mapping
        aux_data["word_ids"] = word_ids

        return tensor_input, aux_data


def main():

    # set config path by command line
    inp_args = utils._parsing()
    config_path = getattr(inp_args, 'yaml')
    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    # print config
    utils._print_config(parameters, config_path)

    entityExtraction = EntityExtraction(config_path)

    text_dir = parameters["test_dir"]
    files = sorted(glob(f"{text_dir}/*.txt"))

    for file in files:
        with open(file) as fp:
            text = fp.read().strip('\n')

        print("###", file, "###")
        sents = sentence_split(text)

        for sent in sents:
            print("-"*10)
            print(sent)
            entities = entityExtraction.get_entities(sent)
            for k, ent in enumerate(entities):
                print(k, ent)


if __name__ == '__main__':
    main()
