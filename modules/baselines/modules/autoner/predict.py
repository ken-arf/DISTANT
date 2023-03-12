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


import gensim
from tqdm.auto import tqdm

from utils import utils

from model import AutoNER 

from make_dataset import tokenize
from make_dataset import sentence_split

import dataclasses

import pdb


@dataclasses.dataclass
class Entity:
    text: str
    name: str
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
        handler1.setFormatter(logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

        self.logger.addHandler(handler1)

        self.nlp = spacy.load("en_core_sci_sm")
        self.nlp.add_pipe("sentencizer")

        with open(config_yaml, 'r') as stream:
            params = utils._ordered_load(stream)
            self.params = params

        self.load_model()

    def load_model(self):

        if torch.cuda.is_available() and self.params['gpu'] >= 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        word2vec_model = self.params["word2vec"]
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True)

        self.model = AutoNER(self.params, self.logger)

        model_path = self.params["restore_model_path"]
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

    def get_entities(self, text):

        self.entity_types = self.params["entity_names"]
        self.ent2id = {ent:k for k, ent in enumerate(self.entity_types)}
        self.id2ent = {k:ent for k, ent in enumerate(self.entity_types)}

        entities = self._get_entities_helper(text)

        return entities

    def _get_entities_helper(self, text):

        entities = []
        tokenized_input, aux_data = self.tokenize_text(text)
        pred_spans, pred_ys = self.model.predict(**tokenized_input)

        for span, y in zip(pred_spans, pred_ys):
            entity_name = self.id2ent[y]

            tokens = aux_data["tokens"][span[0]:span[1]]
            offsets = aux_data["token_offsets"][span[0]:span[1]]

            start_char = offsets[0]
            end_char = offsets[-1] + len(tokens[-1])
            mention = text[start_char:end_char]

            ent = Entity(text=mention, name=entity_name, start=offsets[0], end=offsets[-1], start_char=start_char, end_char=end_char)
            entities.append(ent)

        return entities

    def get_entities_old(self, text):

        entities = self.params["entity_names"]
        ent2id = {}
        ent2id['O'] = 0
        for entity in entities:
            ent2id[f'B-{entity}'] = len(ent2id)
            ent2id[f'I-{entity}'] = len(ent2id)

        all_entities = []
        for entity in entities:
            eid = ent2id[f"B-{entity}"]

            #pdb.set_trace()
            entities = self._get_entities_helper(text, eid, entity)
            print(entities)
            all_entities += entities

        all_entities = sorted(all_entities, key = lambda x: x.start_char)
        return all_entities

    def _get_entities_helper_old(self, text, start_bio, entity_name):

        #pdb.set_trace()
        tokenized_input, aux_data = self.tokenize_text(text)
        prediction, prob = self.model.predict(**tokenized_input)

        # extract location of start_bio 
        indexes = np.argwhere(prediction[0]==start_bio).squeeze(-1).tolist()
        entity_indexes = []

        N = len(prediction[0])
        for index in indexes:

            span = []
            span.append(index)
            for i in range(index+1, N):
                if prediction[0][i] == start_bio+1:
                    span.append(i)
                else:
                    entity_indexes.append(span.copy())
                    break

        # conver index to word_ids
        word_indexes = [[aux_data["word_ids"][index]for index in span] for span in entity_indexes]

        entities = []
        for span_index in word_indexes:

            try:

                # start word index
                start = span_index[0]
                # end word index
                end = span_index[-1]+1

                tokens = aux_data["tokens"][start:end]
                offsets = aux_data["token_offsets"][start:end]
                start_char = offsets[0]
                end_char = offsets[-1] + len(tokens[-1])
                text = aux_data["text"][start_char:end_char]
                ent = Entity(text=text, name=entity_name, start=start, end=end, start_char=start_char, end_char=end_char)

                if not ent in entities:
                    entities.append(ent)

            except:
                print("error", span_index)
                pass

        return entities
            
    def tokenize_text(self, text):

        tokens = tokenize(text, offset=True)

        tokenized_text = [token[0] for token in tokens]
        token_offset = [token[1] for token in tokens]

        tokenized_input = {}

        UNK_id = self.w2v_model.get_index('UNK')
        input_ids = [self.w2v_model.get_index(word.lower()) if word.lower() in self.w2v_model else UNK_id for word in tokenized_text]

        tokenized_input["input_ids"] = [input_ids]
        tokenized_input["slength"] = [len(input_ids)]

        input_char_ids = []
        for word in tokenized_text:
            char_ids = [ord(ch) for ch in list(word.lower())]
            input_char_ids.append(char_ids)

        # compute maximum char-length within the mini-batch
        max_char_len = 0
        for input_char_id in input_char_ids:
            char_len = len(input_char_id)
            if char_len > max_char_len:
                max_char_len = char_len 
    

        tokenized_input['input_char_ids'] = []
        for input_char_id in input_char_ids:
            padded_char_id = input_char_id + [-100] * (max_char_len - len(input_char_id)) 
            tokenized_input['input_char_ids'].append(padded_char_id)
        tokenized_input['input_char_lengths'] = [len(input_char_id) for input_char_id in input_char_ids]

        # add batch dimention
        tokenized_input['input_char_ids'] = [tokenized_input['input_char_ids']]
        tokenized_input['input_char_lengths'] = [tokenized_input['input_char_lengths']]

        tensor_input = {k: torch.tensor(v, dtype=torch.int64).to(self.device) for k, v in tokenized_input.items()}
            
        aux_data = {}
        aux_data["tokens"] = tokenized_text
        aux_data["token_offsets"] = token_offset

        return tensor_input, aux_data 


    def tokenize_text_old(self, text):

        tokens = tokenize(text, offset=True)

        token_text = [token[0] for token in tokens]
        token_offset = [token[1] for token in tokens]

        if len(tokens) > 512:
            print("Error, token length exceed 512")
            exit()

        tokenized_inputs = self.tokenizer(
                    [token_text],
                    truncation = True,
                    max_length = 512,
                    is_split_into_words = True,
                    #return_tensors = "pt",
                    return_offsets_mapping = True,
                    return_overflowing_tokens = True,
        )

        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_inputs.pop("offset_mapping")
        word_ids = tokenized_inputs.word_ids(0)

        tensor_input = {k: torch.tensor(v, dtype=torch.int64).to(self.device) for k, v in tokenized_inputs.items()}
         
        aux_data = {}
        aux_data["text"] = text
        aux_data["tokens"] = token_text
        aux_data["token_offsets"] = token_offset
        aux_data["offset_mapping"] = offset_mapping
        aux_data["word_ids"] = word_ids

        return tensor_input, aux_data 

def output_annotation_file(doc_file, output_dir, extracted_entities):

    _, fname = os.path.split(doc_file)
    basename, ext  = os.path.splitext(fname)

    with open(doc_file) as fp:
        txt = fp.read()
    doc_len = len(txt)

    ann_file_path = os.path.join(output_dir, f"{basename}.ann")

    with open(ann_file_path, 'w') as fp:

        k = 0
        for offset, entities in extracted_entities:
            for entity in entities:
            
                mention = entity.text
                entity_type = entity.name

                start_char = entity.start_char
                end_char = entity.end_char
                start_char += offset
                end_char += offset


                # To prevent an error by brat
                # The last character is "\n", which should not be included in the span definition.
                if end_char == doc_len:
                    end_char -= 1
                
                #assert(txt[start_char:end_char] == mention)

                fp.write(f"T{k+1}\t{entity_type} {start_char} {end_char}\t{mention}\n")
                k += 1

def main():

    # set config path by command line
    inp_args = utils._parsing()                                                                                            
    config_path = getattr(inp_args, 'yaml')
    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    # print config
    utils._print_config(parameters, config_path)

    entityExtraction  = EntityExtraction(config_path)

    text_dir=parameters["test_dir"]
    files = sorted(glob(f"{text_dir}/*.txt"))
    
    output_dir = parameters["output_dir"]
    utils.makedir(output_dir)


    for file in files:
        with open(file) as fp:
            text = fp.read().strip('\n')

        print("###", file, "###")
        sents, offsets = sentence_split(text)

        extracted_entities = []
        for sent, offset in zip(sents, offsets):
            print("-"*10)
            print(sent)
            entities = entityExtraction.get_entities(sent)
            for k, ent in enumerate(entities):
                print(k, ent)
            extracted_entities.append((offset, entities))
        
        output_annotation_file(file, output_dir, extracted_entities)

if __name__ == '__main__':                                                                                                                        
    main()


