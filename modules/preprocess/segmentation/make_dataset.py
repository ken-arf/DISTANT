#!/usr/bin/env python
# coding: utf-8


import sys
import os
import glob
import spacy
import scispacy
import time
from collections import defaultdict
from tqdm import tqdm

from utils import utils
from utils import moses

import pdb

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("sentencizer")

mymoses = moses.MyMosesTokenizer()

def sentence_split(doc, offset = False, moses = False):
    
    if moses:
        sents = mymoses.split_sentence(doc.strip())
        sents = [sent.text for sent in sents]
    else:
        doc = nlp(doc)
        sents =  [sent.text for sent in doc.sents]
        
    return sents
    

def tokenize(text, offset = False, moses = False, ):
    
    if moses:
        tokens = mymoses.tokenize(text.strip())
        tokens = [token.text for token in tokens]
    else:
        doc = nlp(text)


        if offset == False:
            tokens = [token.text for token in doc]
        else:
            tokens = [(token.text, token.idx) for token in doc]
            

    return tokens 
    

def load_dict(path):
    items = []
    with open(path) as fp:
        lines = [line.strip().lower() for line in fp.readlines() if len(line.strip()) !=0]
    for line in lines:
        doc = tokenize(line)
        #print(doc)
        items.append(doc)
    return items
    

def match_entity(tokens, entity_dict, entity_type):
    #print(tokens)
    n = len(tokens)
    bio = ['O'] * n
    for i in range(len(tokens)):
        for entity in entity_dict[entity_type]:
            if i+len(entity) > n-1:
                continue
            tgt = tokens[i:i+len(entity)]
            if tuple(tgt) == tuple(entity):
                #print('match at {}: {}'.format(i, '_'.join(entity)))

                if len(entity) == 1:
                    bio[i] = f'S_{entity_type}'
                else: 
                    for j in range(i, i+len(entity)):
                        if j == i:
                            bio[j] = f'B_{entity_type}'
                        else:
                            bio[j] = f'I_{entity_type}'
            
    return bio
        

def annotate(files, parameters):

    outdir = parameters["output_dir"]
    dict_dirs = parameters["dict_dir"]
    dict_files= parameters["dict_files"]

    dict_paths = [os.path.join(dict_dir, file) for dict_dir in dict_dirs for file in dict_files ]
    print(dict_paths)

    entity_dict = defaultdict(list)

    for dict_path in dict_paths:
        path, fname = os.path.split(dict_path)
        name, txt = os.path.splitext(fname)
        entity_dict[name] += load_dict(dict_path)


    for key, items in entity_dict.items():
        entity_dict[key] = list(set([tuple(item) for item in items]))


    for key in entity_dict.keys():
        entity_dict[key] = sorted(entity_dict[key], key =lambda x: len(x))


    for file in tqdm(files):
        
        path, fname = os.path.split(file)
        with open(file) as fp:
                text = fp.read().strip()
        #print(text)
        doc = sentence_split(text)
        
        with open(os.path.join(outdir, fname), 'w') as fp:
            for k, sent in enumerate(doc):
                tokens = tokenize(sent)
                tokens_low = [token.lower() for token in tokens]
                
                bio_tag = {}
                for entity_type in entity_dict.keys():
                    bio_tag[entity_type] = match_entity(tokens_low, entity_dict, entity_type)
                    
                #print(bio_tag)
                labels = [tags for tags in zip(*bio_tag.values())]
                    
                for token, bio_label in zip(tokens, labels):
                    buf = "{}\t{}".format(token, token.lower())
                    
                    for tag in bio_label:
                        buf +=  "\t{}".format(tag)
                    #print(buf)
                    fp.write("{}\n".format(buf))    
                fp.write('\n')
                

    
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

    corpus_dir = parameters["corpus_dir"]
    output_dir = parameters["output_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob.glob(f"{corpus_dir}/*")

    annotate(files, parameters)

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':                                                                                                                        
    main()


