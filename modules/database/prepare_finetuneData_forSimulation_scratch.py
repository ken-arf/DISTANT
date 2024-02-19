
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
#from elasticsearch import Elasticsearch, helpers

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


def load_brad_annotation(file, term_only = True):
    
    #print(file)

    fnames = []
    tnames = []
    etypes = []
    charStarts = []
    charEnds = []
    mentions = []

    _, fname = os.path.split(file)
    with open(file) as fp:
        for line in fp:
            if line.startswith('T'):
                fields = line.strip().split()
            
                try:
                    tname = fields[0]
                    etype = fields[1]
                    charStart = int(fields[2])
                    charEnd = int(fields[3])
                    mention = fields[4]
        
                    fnames.append(fname)
                    tnames.append(tname)
                    etypes.append(etype)
                    charStarts.append(charStart)
                    charEnds.append(charEnd)
                    mentions.append(mention)
                except:
                    print("Exception")
                    print(fields)
                    print("Do no support flagmented NE")


    df = pd.DataFrame({'fname': fnames,
                        'tname': tnames,
                        'etype': etypes,
                        'charStart': charStarts,
                        'charEnd': charEnds,
                        'mention': mentions})


    return df
            


def simulate_user_update(params):


    gold_annotation_dir = params["gold_annotation_dir"]
    annotation_dir = params["annotation_dir"]

    anns = sorted(glob(f"{gold_annotation_dir}/*.ann"))

    dfs = []
    for ann in anns:
        df = load_brad_annotation(ann)
        dfs.append(df)

    df_gold = pd.concat(dfs, axis=0)


    anns = sorted(glob(f"{annotation_dir}/*.ann"))

    dfs = []
    for ann in anns:
        df = load_brad_annotation(ann)
        dfs.append(df)

    df_ds = pd.concat(dfs, axis=0)

    # delete all rows #####################
    df_ds.drop(df_ds.index,inplace=True) 
    ########################################

    df_ds_update = user_update(df_ds, df_gold, params)

    return df_ds_update

def user_update(df_ds, df_gold, params):

    gold_sample_ratio = params['gold_sample_ratio']
    random_seed = params['random_seed']

    n = int(gold_sample_ratio * df_gold.shape[0])

    df_gold_shuffle = df_gold.sample(frac=1, random_state=random_seed)

    df_gold_samples = df_gold_shuffle[:n]

    #return df_gold_samples.copy()
    
    df_ds['updated'] = ['False'] *  len(df_ds)

    df_ds_update, del_count  = _update(df_ds, df_gold_samples)
    df_ds_update = _delete(df_ds_update, df_gold, del_count, random_seed)


    mask = df_ds_update['updated'].isin(['True', 'Delete'])
    updated_anns = sorted(list(set(df_ds_update[mask]['fname'].tolist())))

    mask = df_ds_update['fname'].isin(updated_anns)
    df_ds_update = df_ds_update[mask]


    return df_ds_update

    
def _delete(df_ds_update, df_gold, del_count, random_seed):

    
    df_ds_update_copy = df_ds_update.copy()
    
    df_ds_update_ = df_ds_update[df_ds_update['updated'] == 'False']


    df_ds_update_shuffle_ = df_ds_update_.sample(frac=1, random_state=random_seed)

    index_for_del = []
    for i, row in df_ds_update_shuffle_.iterrows():
        
        fname = row.fname
        tname = row.tname
        charStart = int(row.charStart)
        charEnd = int(row.charEnd)
        etype = row.etype
        mention = row.mention

        df_gold_ = df_gold[df_gold['fname'] == fname]
        
        ne_spans = [t for t in zip(df_gold_['charStart'].tolist(), df_gold_['charEnd'].tolist())]
    
        (op, offset) = _deside_operation((charStart, charEnd), ne_spans)

        if op == 'NEW':
            index_for_del.append(i)
            #mask = df_ds_update_copy['fname'] == fname
            #df_ds_update_copy.loc[mask, 'updated'] = True
            if len(index_for_del) >= del_count:
                break

    if len(index_for_del) < del_count:
        print(f"** Warning, not enough data to delete **")
        print(f"del_count: {del_count}, deleted_count: {len(index_for_del)}")

    #df_ds_update_copy.drop(index_for_del, inplace = True)

    df_ds_update_copy.loc[index_for_del, 'updated'] = 'Delete'

    return df_ds_update_copy

    
def _update(df_ds, df_gold_samples):

    df_ds.reset_index(inplace=True)

    df_ds_copy = df_ds.copy()


    del_count = 0
    for i, row in df_gold_samples.iterrows():
        #print(row)

        fname = row.fname
        tname = row.tname
        charStart = row.charStart
        charEnd = row.charEnd
        etype = row.etype
        mention = row.mention


        df_ds_ = df_ds[df_ds['fname'] == fname]

        ne_spans = [t for t in zip(df_ds_['charStart'].tolist(), df_ds_['charEnd'].tolist())]

        (op, offset) = _deside_operation((charStart, charEnd), ne_spans)


        new_frame = pd.DataFrame({  'fname': [fname],
                                    'tname': [tname],
                                    'charStart': [charStart],
                                    'charEnd': [charEnd],
                                    'etype': [etype],
                                    'mention': [mention],
                                    'updated': ['True']})

        if op == 'NEW':
            df_ds_copy = pd.concat([df_ds_copy, new_frame], axis=0)
        elif op == 'EXACT':
            row_orig = df_ds_.iloc[offset]
            #print(f'{row_orig}')
            #print(f'{row}')
            assert(row_orig.mention == row.mention)
            index = df_ds_.index[offset]
            if row_orig.etype != etype:
                df_ds_copy.loc[index, 'etype' ] = etype 
                df_ds_copy.loc[index, 'updated' ] = 'True'
            else:
                del_count += 1

        elif op == 'OVERLAP':
            index = df_ds_.index[offset]
            df_ds_copy.loc[index, 'charStart'] = charStart
            df_ds_copy.loc[index, 'charEnd'] = charEnd
            df_ds_copy.loc[index, 'etype'] = etype
            df_ds_copy.loc[index, 'mention'] = mention
            df_ds_copy.loc[index, 'updated'] = 'True'
            

    df_ds_copy.reset_index(inplace=True, drop = True)
    
    return df_ds_copy, del_count


def _deside_operation(gold_span, ne_spans):


    gspan_start_index = -1
    gspan_end_index = -1

    for k, span in enumerate(ne_spans):
        if gold_span[0] >= span[0] and gold_span[0] <= span[1]:
            gspan_start_index = k
        if gold_span[1] >= span[0] and gold_span[1] <= span[1]:
            gspan_end_index = k

    target_k = -1
    if gspan_start_index == -1 and gspan_end_index == -1:
        op = 'NEW'
    elif gspan_start_index != -1 or gspan_end_index != -1:
        op = 'OVERLAP'
        target_k = gspan_start_index if gspan_start_index != -1 else gspan_end_index
        if gspan_start_index == gspan_end_index:
            if int(ne_spans[target_k][0]) == int(gold_span[0]) and int(ne_spans[target_k][1]) == int(gold_span[1]):
                op = 'EXACT'
            
    #print(f"{gspan_start_index}")
    #print(f"{gspan_end_index}")

    return (op, target_k)


def generate_finetune_annotation(df_update, params):


    ann_files = pd.unique(df_update["fname"])
    n = len(ann_files)

    text_dir = params['text_dir']

    for k, ann_file in enumerate(ann_files):
        print(f"{k}/{n} generate conll annotaton for {ann_file}")

        pmid, _ = os.path.splitext(ann_file)
        
        with open(os.path.join(text_dir, f'{pmid}.txt')) as fp:
            text = fp.read()

        #print(text)
        df = df_update[df_update['fname'] == ann_file]
        df_sorted = df.sort_values(by = ['charStart'])
        #print(f"{k+1}/{n}: {ann_file}")
        #print(df_sorted)
        
        entities = []
        for i, row in df_sorted.iterrows():
            entity = {}
            entity['start_char'] = int(row.charStart)
            entity['end_char'] = int(row.charEnd)
            entity['entityType'] = row.etype
            entity['mention'] = row.mention
            entity['updated'] = row.updated
            entities.append(entity)

        annotate_conll(pmid, text, entities, params)
        annotate_span(pmid, text, entities, params)


def annotate_conll(pmid, text, entities, params):


    label_weight_update = params['label_weight_update']

    label2int = {'O': 0, 'B': 1, 'I': 2, 'S':3}

    conll_dir = params['conll_dir']
    utils.make_dirs(conll_dir)

    doc = sentence_split(text, offset = True)


    with open(os.path.join(conll_dir, f'{pmid}.txt'), 'w') as fp:

        insert_annotation = False
        for k, sent in enumerate(doc):

            offset = sent[1]
            tokens = tokenize(sent[0], offset = True)
            tokens_ = [token[0]  for token in tokens]
            tokens_low = [token[0].lower()  for token in tokens]
            tokens_offset = [token[1] + offset for token in tokens]
            bio_labels = ['O'] * len(tokens_low)
            label_weights = [1] * len(tokens_low)

            for entity in entities:

                s_char = entity['start_char']
                e_char = entity['end_char']
                if s_char in tokens_offset:

                    # if updated flag is 'Delete', cancel BIO tags
                    if entity['updated'] == 'Delete' or entity['updated'] == 'True':
                        weight = label_weight_update 
                    else:
                        weight = 1 


                    s_index = tokens_offset.index(s_char)
                    for i in range(s_index, len(tokens_offset)):
                        if tokens_offset[i] >= e_char:
                            break
                    e_index = i - 1
    
                    
                    if s_index == e_index:
                        bio_labels[s_index] = 'S' if entity['updated'] != 'Delete' else 'O'
                    else:
                        bio_labels[s_index] = 'B' if entity['updated'] != 'Delete' else 'O'
                        for i in range(s_index + 1, e_index + 1):
                            bio_labels[i] = 'I' if entity['updated'] != 'Delete' else 'O'

                    label_weights[s_index: e_index+1] = [weight] * (e_index - s_index + 1)


            for t, token, off, w, l in zip(tokens_, tokens_low, tokens_offset, label_weights, bio_labels):
                fp.write(f'{t}\t{token}\t{off}\t{w}\t{l}\n')
            fp.write('\n')

    

def annotate_span(pmid, text, entities, params):


    label_weight_update = params['label_weight_update']

    span_dir = params['span_dir']
    entity_classes = params['entity_names']

    class2id = {c:i for i, c in enumerate(entity_classes)}
    id2class = {i:c for i, c in enumerate(entity_classes)}


    doc = sentence_split(text, offset = True)

    dataset = {
        'pmid': [],
        'start_chars': [],
        'end_chars': [],
        'text': [],
        'entity': [],
        'weight': [],
        'label': [],
    }


    negatives = []
    
    for k, sent in enumerate(doc):

        sentence = sent[0]
        offset = sent[1]
        
        tokens = tokenize(sentence, offset = True)
        subwords = [token[0]  for token in tokens]
        offsets = [token[1] + offset for token in tokens]

        #negatives = sample_negative_span(sentence, tokens)

        for entity in entities:

            #if entity['updated'] == 'Delete':
            #    continue

            if entity['updated'] == 'True' or entity['updated'] == 'Delete':
                weight = label_weight_update
            else:
                weight = 1

            s_char = entity['start_char']
            e_char = entity['end_char']
            if s_char >= offset and e_char <= offset + len(sent[0]):
                mention = entity['mention']
                class_name = entity['entityType']
                s_char -= offset
                e_char -= offset

                ### TODO new entity type
                try:
                    label = class2id[class_name] if entity['updated'] != 'Delete' else len(class2id)
                    #label = class2id[class_name]

                    assert sentence[s_char:e_char] == mention, f"{sentence[s_char:e_char]} not match with {mention}"

                    dataset['pmid'].append(pmid)
                    dataset['start_chars'].append(s_char)
                    dataset['end_chars'].append(e_char)
                    dataset['text'].append(sentence)
                    dataset['entity'].append(mention)
                    dataset['label'].append(label)
                    dataset['weight'].append(weight)

                    #filter(lambda x: x['s_char'] != s_char and x['e_char'] != e_char, negatives)
                except:
                    pass

        # append negative samples

        for negative in negatives:
            #print(negative)

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


    gold_sample_ratio = parameters["gold_sample_ratio"]

    df_update = simulate_user_update(parameters)

    sim_dir = parameters['simulation_dir']
    utils.make_dirs(sim_dir)
    df_update.to_csv(os.path.join(sim_dir, "simulation_data.csv"))

    generate_finetune_annotation(df_update, parameters)
    
    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))

if __name__ == '__main__':                                                                                                                        
    main()