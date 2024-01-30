
import os
import sys
import requests
import argparse
import random
import time
from datetime import datetime
from collections import defaultdict
import re
import pickle
import json
from glob import glob
import pandas as pd

import logging
from utils import utils

import spacy
import pdb

from utils import utils
from elasticsearch import Elasticsearch, helpers

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("sentencizer")

def sentence_split(doc, offset=False):

    doc = nlp(doc)
    if not offset:
        sents = [sent.text for sent in doc.sents]
    else:   
        sents = [(sent.text, sent.start_char) for sent in doc.sents]

    return sents


def tokenize(text, offset=False):

    doc = nlp(text)

    if offset == False:
        # tokens = [token.text for token in doc]
        tokens = [token.text for token in doc if token.text != '\n']
    else:
        # tokens = [(token.text, token.idx) for token in doc]
        tokens = [(token.text, token.idx)
                  for token in doc if token.text != '\n']

    return tokens

def process_documents(params, target, count):

    if target == 'updated':
        file = params['pubmed_updated_documents']
    else:
        file = params['pubmed_unchanged_documents']

    extract_dir = params['pubmed_extract_dir']
    conll_dir = params['pubmed_conll_dir']

    with open(file) as fp:
        data = json.load(fp)

    pmids = []
    try:
        hits = data['hits']['hits']

        random.shuffle(hits)

        if count != -1:
            hits = hits[:count]

        if len(hits) == 0:
            return False

        for index, hit in enumerate(hits):
            pmid = hit['_source']['pmid']
            text = hit['_source']['text']
            entities = hit['_source']['entities']
            
            fname = os.path.join(extract_dir, f"{pmid}.txt")
            with open(fname, 'w') as fp:
                fp.write(f'{text}')

            annotate_conll(pmid, text, entities, params)
            annotate_span(pmid, text, entities, params)
            pmids.append(pmid)

    except:
        print("exception")
        raise

    path, fname = os.path.split(file)
    basename, _ = os.path.splitext(fname)

    with open(os.path.join(path, f"{basename}.pmid"), "w") as fp:
        fp.write('\n'.join(pmids))


    return pmids

def annotate_conll(pmid, text, entities, params):

    label2int = {'O': 0, 'B': 1, 'I': 2, 'S':3}

    conll_dir = params['pubmed_conll_dir']

    doc = sentence_split(text, offset = True)


    with open(os.path.join(conll_dir, f'{pmid}.txt'), 'w') as fp:
        for k, sent in enumerate(doc):

            offset = sent[1]
            tokens = tokenize(sent[0], offset = True)
            tokens_ = [token[0]  for token in tokens]
            tokens_low = [token[0].lower()  for token in tokens]
            tokens_offset = [token[1] + offset for token in tokens]
            bio_labels = ['O'] * len(tokens_low)

            for entity in entities:
                s_char = entity['start_char']
                e_char = entity['end_char']
                if s_char in tokens_offset:
                    print(entity)
                    s_index = tokens_offset.index(s_char)
                    for i in range(s_index, len(tokens_offset)):
                        if tokens_offset[i] >= e_char:
                            break
                    e_index = i - 1
    
                    if s_index == e_index:
                        bio_labels[s_index] = 'S'
                    else:
                        bio_labels[s_index] = 'B'
                        for i in range(s_index + 1, e_index + 1):
                            bio_labels[i] = 'I'

            for t, token, off, l in zip(tokens_, tokens_low, tokens_offset, bio_labels):
                print(f'{t}\t{token}\t{off}\t{l}')
                fp.write(f'{t}\t{token}\t{off}\t{l}\n')
            print('')
            fp.write('\n')

def annotate_span(pmid, text, entities, params):


    span_dir = params['pubmed_span_dir']
    #entity_classes = params['entity_names']
    entity_classes = get_entity_types(params)

    class2id = {c:i for i, c in enumerate(entity_classes)}
    id2class = {i:c for i, c in enumerate(entity_classes)}


    doc = sentence_split(text, offset = True)

    dataset = {
        'pmid': [],
        'start_chars': [],
        'end_chars': [],
        'text': [],
        'entity': [],
        'label': []
    }

    
    for k, sent in enumerate(doc):

        sentence = sent[0]
        offset = sent[1]
        
        tokens = tokenize(sentence, offset = True)
        subwords = [token[0]  for token in tokens]
        offsets = [token[1] + offset for token in tokens]

        negatives = sample_negative_span(sentence, tokens)

        for entity in entities:

            s_char = entity['start_char']
            e_char = entity['end_char']
            if s_char >= offset and e_char <= offset + len(sent[0]):
                mention = entity['mention']
                class_name = entity['entityType']
                s_char -= offset
                e_char -= offset

                ### TODO new entity type
                try:
                    label = class2id[class_name]

                    assert sentence[s_char:e_char] == mention, f"{sentence[s_char:e_char]} not match with {mention}"

                    dataset['pmid'].append(pmid)
                    dataset['start_chars'].append(s_char)
                    dataset['end_chars'].append(e_char)
                    dataset['text'].append(sentence)
                    dataset['entity'].append(mention)
                    dataset['label'].append(label)

                    filter(lambda x: x['s_char'] != s_char and x['e_char'] != e_char, negatives)
                except:
                    pass

        # append negative samples

        for negative in negatives:
            print(negative)

            s_char = negative['s_char']
            e_char = negative['e_char']
            mention = negative['mention']
            label = len(class2id)

            dataset['pmid'].append(pmid)
            dataset['start_chars'].append(s_char)
            dataset['end_chars'].append(e_char)
            dataset['text'].append(sentence)
            dataset['entity'].append(mention)
            dataset['label'].append(label)

    utils.make_dirs(span_dir)
    fname = os.path.join(span_dir, f'{pmid}.csv')
    df = pd.DataFrame.from_dict(dataset)
    df.to_csv(fname)

def sample_negative_span(sentence, tokens):

    neg_samples = []

    max_span_len = 5
    max_samples = 3 

    n = len(tokens)

    k = 0
    while k < max_samples:

        if n == 0:
            break

        print("n=", n)
        start = random.randint(0,n-1)
        l_limit = min(n - start, max_span_len)
        span_len = random.randint(1, l_limit)
        end = start + span_len - 1

        s_char = tokens[start][1]
        e_char = tokens[end][1] + len(tokens[end][0])
        mention= sentence[s_char:e_char]

        neg_samples.append({
                's_char': s_char,
                'e_char': e_char,
                'mention': mention})

        k += 1
    
    return neg_samples


def get_entity_types(params):

    es_addr = params["elastic_search"]
    es_index = params["index_name"]

    es = Elasticsearch(es_addr)
    index=es_index

    resp = es.get(index=index, id="1")
    
    try:
        entity_types = resp['_source']['entity_types']
    except:
        print("database error")
        exit()

    return entity_types



def main():

    #get_abstract(27895423)

    # check running time
    t_start = time.time()

    # logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter(
        "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

    # handler2 = logging.FileHandler(filename="test.log")
    # handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(handler1)

    # set config path by command line
    inp_args = utils._parsing()
    config_path = getattr(inp_args, 'yaml')
    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    # print config
    utils._print_config(parameters, config_path)

    pmids_train = process_documents(parameters, 'updated', -1)
    pmids_val = process_documents(parameters, 'upchanged', 100 * len(pmids_train))

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))

if __name__ == '__main__':                                                                                                                        
    main()
