import pandas as pd
import spacy
import scispacy
import os
import time
import pickle
import numpy as np
from collections import defaultdict
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

from entity_extract import ExtractEntityCandidate

from tqdm.auto import tqdm

from utils import utils

from model2 import Model

import pdb


class Entity_extractor:

    def __init__(self, params):

        self.params = params

        # logging
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.ERROR)

        handler1 = logging.StreamHandler()
        handler1.setFormatter(logging.Formatter(
            "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

        # handler2 = logging.FileHandler(filename="test.log")
        # handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

        self.logger.addHandler(handler1)
        # logger.addHandler(handler2)

        # self.nlp = spacy.load("en_core_sci_lg")
        self.nlp = spacy.load("en_core_sci_sm")
        self.nlp.add_pipe("sentencizer")

        self.load_model()

    def load_model(self):

        model_checkpoint = self.params["model_checkpoint"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        if torch.cuda.is_available() and self.params['gpu'] >= 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # number of entity classes + negative label
        self.params["class_num"] = len(self.params["entity_names"]) + 1

        self.pu_model = Model(self.params, self.logger)

        self.pu_model.load_state_dict(torch.load(
            self.params['restore_model_path'], map_location=torch.device(self.device)))

        self.entityExtraction = ExtractEntityCandidate(
            self.params["segmentation_predict_config"])

    def data_collator(self, features):

        batch = self.tokenizer.pad(
            features,
            padding=True,
            # max_length=max_length,
            pad_to_multiple_of=None,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors=None,
        )

        batch = {k: torch.tensor(v, dtype=torch.int64).to(
            self.device) for k, v in batch.items()}
        return batch

    def tokenized_and_align_labels(self, example):

        tokenized_inputs = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            is_split_into_words=False,
            # return_tensors = "pt",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )

        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")

        offset_mapping = tokenized_inputs.pop("offset_mapping")

        start_positions = []
        end_positions = []
        sample_ids = []

        for index, (sample_id, offset) in enumerate(zip(sample_map, offset_mapping)):

            start_char = int(example["start_chars"][sample_id])
            end_char = int(example["end_chars"][sample_id])

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

                elif offset[context_end][1] >= end_char:

                    start_position = context_start
                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_position = idx + 1

            if start_position > end_position:
                print("WARNING start_pos > end_pos")
                exit()

            start_positions.append(start_position)
            end_positions.append(end_position)
            sample_ids.append(sample_id)

        tokenized_inputs["start_positions"] = start_positions
        tokenized_inputs["end_positions"] = end_positions
        tokenized_inputs["sample_ids"] = sample_ids

        return tokenized_inputs

    def predict(self, file_path):

        df_doc, data_loader = self.extract_entity(file_path)
        num_samples = df_doc.shape[0]
        # Evaluation
        self.pu_model.eval()

        sample_ids = []
        predicts = []
        for batch in data_loader:
            with torch.no_grad():
                predictions, probs = self.pu_model.decode(**batch)
            sample_id = batch["sample_ids"].detach().cpu().item()
            predict = predictions[0]

            sample_ids.append(sample_id)
            predicts.append(predict)

        assert (len(sample_ids) == num_samples)
        df_doc["predict"] = predicts
        return df_doc

    def _extract_entity_helper(self, sent, offset):

        entities = []
        start_tokens = []
        end_tokens = []
        start_chars = []
        end_chars = []

        candidates = self.entityExtraction.extract_candiate(
            sent, custom_model=True, scipy_model=False)

        for ent in candidates:

            entities.append(ent.text)
            start_tokens.append(int(ent.start))
            end_tokens.append(int(ent.end))
            start_chars.append(int(ent.start_char))
            end_chars.append(int(ent.end_char))

        df = pd.DataFrame({'entities': entities,
                           'start_tokens': start_tokens,
                           'end_tokens': end_tokens,
                           'start_chars': start_chars,
                           'end_chars': end_chars,
                           'text': [sent] * len(entities),
                           'sentence_offset': offset})

        return df

    def extract_entity(self, document_path):

        dfs = []
        with open(document_path) as fp:
            text = fp.read()
            doc = self.nlp(text)
            for k, sent in enumerate(doc.sents):

                df = self._extract_entity_helper(sent.text, sent.start_char)
                dfs.append(df)

        df_doc = pd.concat(dfs, ignore_index=True)

        test_dataset = Dataset.from_pandas(df_doc)

        test_datasets = DatasetDict({
            "test": test_dataset,
        })

        tokenized_datasets = test_datasets.map(
            self.tokenized_and_align_labels,
            batched=True,
            remove_columns=test_datasets["test"].column_names
        )

        test_dataloader = DataLoader(
            tokenized_datasets["test"],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=1,
        )

        return df_doc, test_dataloader


def output_annotation_file(doc_file, df_result, annotation_root_dir, entity_names, domain_dictionary):

    entity_types = {k: name for k, name in enumerate(entity_names)}

    # CYTOKINE=0
    # TRANSCRIPTION_FACTOR=1
    # T_LYMPHOCYTE=2
    # entity_types = {}
    # entity_types[CYTOKINE] = "Cytokine"
    # entity_types[TRANSCRIPTION_FACTOR] = "Transcription_Factor"
    # entity_types[T_LYMPHOCYTE] =  "T_Cell"

    _, fname = os.path.split(doc_file)
    basename, ext = os.path.splitext(fname)

    with open(doc_file) as fp:
        txt = fp.read()
    doc_len = len(txt)

    ann_file_path = os.path.join(annotation_root_dir, f"{basename}.ann")

    non_entity_label = len(entity_types)
    df_result = df_result[df_result["predict"] != non_entity_label]

    with open(ann_file_path, 'w') as fp:
        for k, (index, row) in enumerate(df_result.iterrows()):
            entity = row["entities"]
            start_char = int(row["start_chars"])
            end_char = int(row["end_chars"])
            sent_offset = int(row["sentence_offset"])
            predict = row["predict"]
            start_char += sent_offset
            end_char += sent_offset

            # To prevent an error by brat
            # The last character is "\n", which should not be included in the span definition.
            if end_char == doc_len:
                end_char -= 1

            entity_type = entity_types[predict]
            domain_dict = domain_dictionary[entity_type.lower()]

            if entity.lower() in domain_dict:
                cui = domain_dict[entity.lower()]
            else:
                cui = "NA"

            fp.write(
                f"T{k+1}\t{entity_type} {start_char} {end_char}\t{entity}\n")
            fp.write(f"A{k+1}\tCUI T{k+1} {cui}\n")


def dict_load(dict_path):

    term2cui = {}
    with open(dict_path) as fp:
        for line in fp:
            atom, term_lc, term_lc_head, cui = line.strip().split('|')
            if term_lc not in term2cui:
                term2cui[term_lc] = cui
            if term_lc_head not in term2cui:
                term2cui[term_lc_head] = cui

    return term2cui


def main():

    # set config path by command line
    inp_args = utils._parsing()
    config_path = getattr(inp_args, 'yaml')
    with open(config_path, 'r') as stream:
        params = utils._ordered_load(stream)

    # print config
    utils._print_config(params, config_path)

    entity_extractor = Entity_extractor(params)

    document_root_dir = params['document_root_dir']
    annotation_root_dir = params['annotation_root_dir']

    utils.makedir(annotation_root_dir)

    entity_names = params['entity_names']

    files = sorted(glob(f"{document_root_dir}/*.txt"))

    # load domain_dict
    domain_dictionary = defaultdict(dict)
    dict_dirs = params["processed_dict_dirs"]
    dict_files = params["dict_files"]
    for dict_dir in dict_dirs:
        for dfile in dict_files:
            dict_path = os.path.join(dict_dir, dfile)
            base, _ = os.path.splitext(dfile)
            domain_dictionary[base].update(dict_load(dict_path))

    for file in files:
        print(file)
        df_result = entity_extractor.predict(file)
        output_annotation_file(
            file, df_result, annotation_root_dir, entity_names, domain_dictionary)

    # check running time
    t_start = time.time()
    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()
