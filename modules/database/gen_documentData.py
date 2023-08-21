
import os
import sys
import requests
import argparse
import time
from datemath import datetime
import re
from collections import defaultdict
import pickle
import json
from glob import glob

import logging

from utils import utils

import pdb


def gen_AbstractJsonData(txt, ann, pmid):

    json_data = {}

    with open(txt) as fp:
        txt = fp.read()

    with open(ann) as fp:
        lines = fp.readlines()
    lines = [line.strip() for line in lines]


    entity_list = []
    for i in range(0,len(lines), 2):
        entity = {}

        ent = lines[i]
        atr = lines[i+1]

        # process ent
        fields = ent.split('\t')
        tname = fields[0]
        mention = fields[2]
        ent_type, start_char, end_char = fields[1].split(' ')

        # process atr
        fields = atr.split('\t')
        aname = fields[0]
        _, tname_, cui = fields[1].split(' ')

        entity['entityType'] = ent_type
        entity['mention'] = mention
        entity['start_char'] = int(start_char)
        entity['end_char'] = int(end_char)
        entity['cui'] = cui

        now = datetime.now()
        entity['last_modified'] = now.strftime("%Y-%m-%dT%H:%M:%S")

        entity_list.append(entity)

    json_data['pmid'] = pmid
    json_data['text'] = txt
    json_data['entities'] = entity_list

    index_data = {}
    index_data['_index'] = "abstract"
    index_data['_type'] = "cancer_immunology"
    index_data['_id'] = pmid

    return json_data, index_data 

def gen_EntityJsonData(txt, ann, pmid):

    json_data = {}

    with open(txt) as fp:
        txt = fp.read()

    with open(ann) as fp:
        lines = fp.readlines()
    lines = [line.strip() for line in lines]


    entity_list = []
    for i in range(0,len(lines), 2):
        entity = {}

        ent = lines[i]
        atr = lines[i+1]

        # process ent
        fields = ent.split('\t')
        tname = fields[0]
        mention = fields[2]
        ent_type, start_char, end_char = fields[1].split(' ')

        # process atr
        fields = atr.split('\t')
        aname = fields[0]
        _, tname_, cui = fields[1].split(' ')

        entity['entityType'] = ent_type
        entity['mention'] = mention
        entity['cui'] = cui

        now = datetime.now()
        entity['last_modified'] = now.strftime("%Y-%m-%dT%H:%M:%S")
    
        entity_list.append(entity)

    json_data['entities'] = entity_list

    index_data = {}
    index_data['_index'] = "entity"
    index_data['_type'] = "cancer_immunology"
    index_data['_id'] = pmid

    return json_data, index_data 


def main():

    # check running time                                                                                                   
    t_start = time.time()                                                                                                  

    # logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

    #handler2 = logging.FileHandler(filename="test.log")
    #handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(handler1)

    # set config path by command line
    inp_args = utils._parsing()                                                                                            
    config_path = getattr(inp_args, 'yaml')                                                                                
    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    # print config
    utils._print_config(parameters, config_path)

    extract_dir = parameters["pubmed_extract_dir"]
    annotate_dir = parameters["pubmed_annotate_dir"]

    txt_files = sorted(glob(f"{extract_dir}/*txt"))
    ann_files = sorted(glob(f"{annotate_dir}/*ann"))


    # abstract index

    json_dataset = []
    for txt, ann in zip(txt_files, ann_files):
        path, txt_fname = os.path.split(txt)
        path, ann_fname = os.path.split(ann)
        txt_basename, _  = os.path.splitext(txt_fname)
        ann_basename, _  = os.path.splitext(ann_fname)
        assert(txt_basename == ann_basename)
        print(txt, ann)
        json_data, index_data = gen_AbstractJsonData(txt, ann, ann_basename)
        json_dataset.append((json_data, index_data))

    output_dir = parameters["output_dir"]
    with open(os.path.join(output_dir, 'abstract.jsonl'), 'w')  as fout:
        for json_data, index_data in json_dataset:
            json.dump({'index': index_data} , fout)
            fout.write('\n')
            json.dump(json_data, fout)
            fout.write('\n')

    # entity index

    json_dataset = []
    for txt, ann in zip(txt_files, ann_files):
        path, txt_fname = os.path.split(txt)
        path, ann_fname = os.path.split(ann)
        txt_basename, _  = os.path.splitext(txt_fname)
        ann_basename, _  = os.path.splitext(ann_fname)
        assert(txt_basename == ann_basename)
        print(txt, ann)
        json_data, index_data = gen_EntityJsonData(txt, ann, ann_basename)
        json_dataset.append((json_data, index_data))

    output_dir = parameters["output_dir"]
    with open(os.path.join(output_dir, 'entity.jsonl'), 'w')  as fout:
        for json_data, index_data in json_dataset:
            json.dump({'index': index_data} , fout)
            fout.write('\n')
            json.dump(json_data, fout)
            fout.write('\n')


    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))

if __name__ == '__main__':                                                                                                                        
    main()




