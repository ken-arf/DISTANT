#!/usr/bin/env python
# coding: utf-8


import sys
import os
import re
import glob
import spacy
import scispacy
import time
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing import Process, current_process

from tqdm import tqdm

from utils import utils
from utils import moses

import pdb


# nlp = spacy.load("en_core_sci_lg")
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("sentencizer")

mymoses = moses.MyMosesTokenizer()


def sentence_split(doc, offset=False, moses=False):

    if moses:
        sents = mymoses.split_sentence(doc.strip())
        sents = [sent.text for sent in sents]
    else:

        doc = nlp(doc)
        sents = [sent.text for sent in doc.sents]

    return sents


def tokenize(text, offset=False, moses=False, ):

    if moses:
        tokens = mymoses.tokenize(text.strip())
        tokens = [token.text for token in tokens]
    else:
        doc = nlp(text)

        if offset == False:
            # tokens = [token.text for token in doc]
            tokens = [token.text for token in doc if token.text != '\n']
        else:
            # tokens = [(token.text, token.idx) for token in doc]
            tokens = [(token.text, token.idx)
                      for token in doc if token.text != '\n']


    return tokens

def tokenize_pos(text, offset=False, moses=False, ):

    if moses:
        tokens = mymoses.tokenize(text.strip())
        tokens = [token.text for token in tokens]
    else:
        doc = nlp(text)

        if offset == False:
            # tokens = [token.text for token in doc]
            tokens = [token.text for token in doc if token.text != '\n']
        else:
            # tokens = [(token.text, token.idx) for token in doc]
            tokens = [(token.text, token.idx)
                      for token in doc if token.text != '\n']

        pos = [token.pos_ for token in doc if token.text != '\n']

    return tokens, pos


def load_dict(path):

    items = []
    with open(path) as fp:
        entries = [line.strip().split('|')
                   for line in fp.readlines() if len(line.strip()) != 0]

    for entry in tqdm(entries):
        term = entry[0]
        term_lc = tokenize(entry[1])
        cui = entry[2]

        # (atom string, list of atom tokens, list of head part of atom tokens, cui)
        items.append((term, term_lc, cui))

#    for entry in tqdm(entries):
#        atom_str = entry[0]
#        tokens_lc = tokenize(entry[1])
#        tokens_lc_head = tokenize(entry[2])
#        cui = entry[3]
#
#        # (atom string, list of atom tokens, list of head part of atom tokens, cui)
#        items.append((atom_str, tokens_lc, tokens_lc_head, cui))

    return items


def match_entity(tokens, pos, entity_dict, entity_type):
    # print(tokens)

    POS = ['ADJ', 'NOUN', 'PROPN', 'PART', 'NUM', 'PRON', 'SYM', 'X', 'PUNCT']

    match_count = 0
    n = len(tokens)
    bio = ['O'] * n

    i = 0
    while i < len(tokens):
        found = False
        for entity in entity_dict[entity_type]:
            if i+len(entity) > n-1:
                continue

            tgt = tokens[i:i+len(entity)]
            pos_ = pos[i:i+len(entity)]


            if tuple(tgt) == tuple(entity):

                pos_check = [True if p in POS else False for p in pos_]

                if not all(pos_check):
                    continue
                
                found = True
                match_count += 1

                if len(entity) == 1:
                    bio[i] = f'S_{entity_type}'
                    i += 1
                else:
                    for j in range(i, i+len(entity)):
                        if j == i:
                            bio[j] = f'B_{entity_type}'
                        else:
                            bio[j] = f'I_{entity_type}'
                    i += len(entity)
                break

        if not found:
            i += 1

    return bio, match_count


def span_annotate(f, entity_dict, outdir):

        processName = current_process().name

        print(f"process {processName} started")

        path, fname = os.path.split(f)
        with open(f) as fp:
            text = fp.read().strip()
        # print(text)
        doc = sentence_split(text)

        with open(os.path.join(outdir, fname), 'w') as fp:
            for k, sent in enumerate(doc):
                tokens, pos = tokenize_pos(sent)
                tokens_low = [token.lower() for token in tokens]

                bio_tag = {}
                for entity_type in entity_dict.keys():
                    bio_tag[entity_type], cnt = match_entity(
                        tokens_low, pos, entity_dict, entity_type)
                    #match_count[entity_type] += cnt

                # print(bio_tag)
                labels = [tags for tags in zip(*bio_tag.values())]

                for token, bio_label in zip(tokens, labels):
                    buf = "{}\t{}".format(token, token.lower())

                    for tag in bio_label:
                        buf += "\t{}".format(tag)
                    # print(buf)
                    fp.write("{}\n".format(buf))
                fp.write('\n')

        print(f"process {processName} finish")


def annotate(files, parameters):

    ent2int = parameters["entity2integer"]
    int2ent = {i: k for k, i in ent2int.items()}

    outdir = parameters["output_dir"]
    dict_dirs = parameters["dict_dir"]
    dict_files = parameters["dict_files"]

    dict_paths = [os.path.join(dict_dir, file)
                  for dict_dir in dict_dirs for file in dict_files]
    print(dict_paths)

    entity_dict = defaultdict(list)

    if parameters["use_dictionary"]:
        for dict_path in dict_paths:
            if not os.path.exists(dict_path):
                continue

            path, fname = os.path.split(dict_path)
            name, txt = os.path.splitext(fname)

            print(f"loading dictionary {dict_path}")
            entity_dict[name] += load_dict(dict_path)


    # convert dictionay item to list to tuple
    for key, items in entity_dict.items():
        l_token_seq = [tuple(item[1]) for item in items]
        # l_token_seq += [tuple(item[2]) for item in items]

        entity_dict[key] = list(set(l_token_seq))

    # sort dictionary items in ascending order
    for key in entity_dict.keys():
        entity_dict[key] = sorted(entity_dict[key], key=lambda x: -len(x))


    N = len(files)
    # number of paralle process
    para = 100 
    n = N // para
    x = list(range(N))

    ll = [x[i::n] for i in range(n)]
    
    for i, l in enumerate(ll):

        print(f'process {i}, {l}')

        processes = []

        fs = [files[i] for i in l]
        for f in fs:

            _, fname = os.path.split(f)
            p = Process(target = span_annotate, args = (f, entity_dict, outdir,), name = fname)
            processes.append(p)

        for p in processes:
            p.start()


        for p in processes:
            p.join()

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

    # global nlp
    # nlp = spacy.load(parameters["spacy_model"])
    # nlp.add_pipe("sentencizer")

    corpus_dir = parameters["corpus_dir"]
    output_dir = parameters["output_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted(glob.glob(f"{corpus_dir}/*.txt"))

    if len(files) == 0:
        print("file not found")
        exit(1)

    annotate(files, parameters)

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()
