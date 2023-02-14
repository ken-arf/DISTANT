#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import torch
import glob
from tqdm import tqdm
from collections import defaultdict

from utils import utils

from dataloader import Dataloader

import pdb


def load_file(file,  n_type):
    
    input_data = defaultdict(list)
    
    with open(file) as fp:
        tokens = []
        bio_labels = defaultdict(list)
        
        for line in fp:
            line = line.strip('\n')
            #print(line, len(line))
            if len(line) == 0:
                input_data['tokens'].append(tokens)
                for i in range(n_type):
                    input_data[f'bio_{i}'].append(bio_labels[i])
                tokens = []
                bio_labels = defaultdict(list)
                continue
            fields = line.split('\t')
            #print(fields)
            assert(len(fields) == n_type + 2)
            tokens.append(fields[0])
            for i in range(n_type):
                bio_labels[i].append(fields[2+i])
            
    return input_data
            

def convert_bio(label):
    if label == 'O':
        return 0
    else:
        return 1


def load_dataset(parameters):

    ent_num = len(parameters["entities"])
    corpus_dir = parameters["corpus_dir"]

    files = sorted(glob.glob(f"{corpus_dir}/*.txt"))

    text_data = []
    label_data = []

    for file in tqdm(files):
        #print(file)
        input_data = load_file(file, ent_num)
        #print(input_data.keys())
        seq_num = len(input_data['tokens'])
        
        for sn in range(seq_num):
            lseq = []
            for i in range(ent_num):
                lseq.append(list(map(convert_bio, input_data[f'bio_{i}'][sn])))
            
            bl = [1 if sum(s) > 0 else 0 for s in zip(*lseq)]
    #        print(bl, len(bl))
    #        print(input_data['tokens'][sn], len(input_data['tokens'][sn]))
            text_data.append(input_data['tokens'][sn])
            label_data.append(bl)

    data = {'text': text_data, 'label': label_data}

    return data
            

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

    # step 0) load dataset
    data = load_dataset(parameters)
    dataloader = Dataloader(data, parameters)
    train_dataloader, valid_dataloader = dataloader.load_data()

    for batch in train_dataloader:
        print(batch)
        break

    # step 1) train model

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))
        

if __name__ == '__main__':                                                                                                                        
    main()


