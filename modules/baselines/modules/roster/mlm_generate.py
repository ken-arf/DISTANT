#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import torch
import logging
import glob
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from utils import utils

import torch

from transformers import AutoTokenizer,BertForMaskedLM

import pdb


def load_file(file,  n_type):

    input_data = defaultdict(list)
    
    with open(file) as fp:
        tokens = []
        bio_labels = []
        
        for line in fp:
            line = line.strip('\n')
            #print(line, len(line))
            if len(line) == 0:
                input_data['tokens'].append(tokens)
                input_data['bio'].append(bio_labels)
                tokens = []
                bio_labels = []
                continue
            fields = line.split('\t')
            #print(fields)
            assert(len(fields) == 3)
            tokens.append(fields[0])
            bio_labels.append(fields[2])
            
    return input_data
            

def load_dataset(parameters):

    ent_num = parameters["num_bio_labels"]
    corpus_dir = parameters["corpus_dir"]

    files = sorted(glob.glob(f"{corpus_dir}/*.txt"))

    file_paths = []
    contents = []


    for file in tqdm(files):
        input_data = load_file(file, ent_num)
        contents.append(input_data)
        file_paths.append(file)

    data = {'paths': file_paths, 'contents': contents}

    return data


def gen_text(content, path, tokenizer, model):


    tokens_list = content["tokens"]
    labels_list = content["bio"]

    new_tokens = []
    for tokens in tokens_list:
        alternative_words = gen_alternative_sent(tokens, tokenizer, model)
        new_tokens.append(alternative_words)

    del content["tokens"]
    content["tokens"] = new_tokens

    return

def gen_alternative_sent(words, tokenizer, model):

    mask_token_id = tokenizer.mask_token_id

    tokenized_inputs = tokenizer(
                words,
                truncation = True,
                max_length = 512,
                is_split_into_words = True,
                return_tensors = "pt",
                return_offsets_mapping = True,
                return_overflowing_tokens = True,
                )

    sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_inputs.pop("offset_mapping")

    assert(len(sample_map)==1)
    mask_thres = 0.15
    for index, (sample_id, offset) in enumerate(zip(sample_map, offset_mapping)):
        
        input_ids = tokenized_inputs['input_ids'][0]
        word_ids = tokenized_inputs.word_ids(index)

        l = input_ids.shape[0]
        rand_seq = torch.rand(l)
        # token[0]: [CLS]
        rand_seq[0] = 1.0
        # token[-1]: [SEP]
        rand_seq[-1] = 1.0

        mask = rand_seq < mask_thres
        input_ids.masked_fill_(mask, mask_token_id)
        
        with torch.no_grad():
            logits = model(**tokenized_inputs).logits
        
        y_index = logits[0][mask].argsort(axis=-1, descending=True)
        n = y_index.shape[0]

        token_ids = []
        for k, idx in enumerate(torch.randint(0,4,(n,))):
            token_id = y_index[k,idx]
            token_ids.append(token_id)

        try:
            input_ids[mask] = torch.tensor(token_ids)
            alternative_words = []
            word_num = len(words)
            for i in range(word_num):
                pos = np.where(np.array(word_ids)==i)[0]
                decoded_word = tokenizer.decode(input_ids[pos[0]:pos[-1]+1], 
                                                skip_special_tokens=True,
                                                cleanup_tokenization_spaces=True)
                word = decoded_word.replace(' ','')
                alternative_words.append(word)
        except:
            alternative_words = words

    tokenized_inputs2 = tokenizer(
                alternative_words,
                truncation = True,
                max_length = 512,
                is_split_into_words = True,
                return_tensors = "pt",
                return_offsets_mapping = True,
                return_overflowing_tokens = True,
                )

    input_ids = tokenized_inputs['input_ids']
    input_ids2 = tokenized_inputs2['input_ids']

    # check if the lengths of subword sequences are equal 
    if input_ids.shape != input_ids2.shape:
        alternative_words = words

    return alternative_words


def mlm_generate(parameters, name_suffix):

    # logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.ERROR)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

    #handler2 = logging.FileHandler(filename="test.log")
    #handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(handler1)
    #logger.addHandler(handler2)

    dataset = load_dataset(parameters)

    tokenizer = AutoTokenizer.from_pretrained(parameters["model_checkpoint"])
    model = BertForMaskedLM.from_pretrained(parameters["model_checkpoint"])

    output_dir = parameters['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        

    pbar = tqdm(total=len(dataset["paths"]))

    for content, path in zip(dataset["contents"], dataset["paths"]):
        gen_text(content, path, tokenizer, model)
        _, fname = os.path.split(path)
        with open(os.path.join(output_dir, fname), 'w') as fp:

            for tokens, labels in zip(content["tokens"], content["bio"]):
        
                assert(len(tokens) == len(labels))
                for word, l in zip(tokens, labels):
                    fp.write('{}\t{}\t{}\n'.format(word, word, l))
                fp.write('\n')
        pbar.update(1)
                

def main():

    # set config path by command line
    inp_args = utils._parsing()                                                                                            
    config_path = getattr(inp_args, 'yaml')
    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    # print config
    utils._print_config(parameters, config_path)

    # check running time
    t_start = time.time()                                                                                                  
    mlm_generate(parameters, "roster")

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))
        

if __name__ == '__main__':                                                                                                                        
    main()


