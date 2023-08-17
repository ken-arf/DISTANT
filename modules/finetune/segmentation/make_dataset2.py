#!/usr/bin/env python
# coding: utf-8


import sys
import os
import glob
import spacy
import scispacy
import time
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from utils import utils
from utils import moses

import pdb


#nlp = spacy.load("en_core_sci_lg")
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("sentencizer")

mymoses = moses.MyMosesTokenizer()

def sentence_split(doc, offset = False, moses = False):
    
    if moses:
        sents = mymoses.split_sentence(doc.strip())
        sents = [sent.text for sent in sents]
    else:

        doc = nlp(doc)
        if offset == False:
            sents =  [sent.text for sent in doc.sents]
        else:
            sents =  [(sent.text, sent.start_char)  for sent in doc.sents]

        
    return sents
    

def tokenize(text, offset = False, moses = False, ):
    
    if moses:
        tokens = mymoses.tokenize(text.strip())
        tokens = [token.text for token in tokens]
    else:
        doc = nlp(text)

        if offset == False:
            #tokens = [token.text for token in doc]
            tokens = [token.text for token in doc if token.text != '\n']
        else:
            #tokens = [(token.text, token.idx) for token in doc]
            tokens = [(token.text, token.idx) for token in doc if token.text != '\n']
            
    return tokens 
    

def annotate_helper_(tokens, entities):

    labels = ["O"] * len(tokens)
    for entity in entities:
        ent_start = entity["start_char"]
        ent_end = entity["end_char"]
        etype = entity["entityType"]
        mention = entity["mention"]

        entity_start = -1
        entity_end = -1
        for i in range(len(tokens)):
            if tokens[i][1] == ent_start:
                entity_start = i
                for j in range(i, len(tokens)):
                    if tokens[j][1] + len(tokens[j][0]) == ent_end:
                        entity_end = j

        print(f"{entity_start}:{entity_end}:{etype}")
        if entity_start != -1 and entity_end != -1:
            labels[entity_start] = f"B-{etype}"
            for i in range(entity_start+1, entity_end+1):
                labels[i] = f"I-{etype}"


    blabels = [1 if l != 'O' else 0 for l in labels]

    print(tokens)
    print(labels)
    print(blabels)
    return labels, blabels

    
def annotate(json_data, parameters):

    ent2int = parameters["entity2integer"]
    int2ent = {i:k for k,i in ent2int.items()}
    

    outdir = parameters["output_dir"]

    ids = json_data["pmid"].keys()


    for id_ in sorted(ids):
        pmid = json_data["pmid"][id_]
        text = json_data["text"][id_]
        entities = json_data["entities"][id_]

        fname = f"{pmid}.txt"
        doc = sentence_split(text.strip(), offset=True)

        print(fname)
        with open(os.path.join(outdir, fname), 'w') as fp:
            for k, (sent, sent_offset) in enumerate(doc):

                print("sent:", sent, len(sent))
                tokens = tokenize(sent, offset=True)
                tokens = [(token, sent_offset + offset_) for token, offset_ in tokens]
    
                labels, blabels = annotate_helper_(tokens, entities)

                for token_, label_, blabel_ in zip(tokens, labels, blabels):
                    fp.write(f"{token_[0]}\t{label_}\t{blabel_}\n")
                fp.write("\n")


    return 

    
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

    json_dump= parameters["es_dump_path"]
    output_dir = parameters["output_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_dump) as fp:
        json_data = json.load(fp)

    match_count = annotate(json_data, parameters)

    print("match_count")
    print(match_count)



    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':                                                                                                                        
    main()


