
import os
import sys
import requests
import argparse
import time
from datetime import datetime
import re
from collections import defaultdict
import pickle
import json
from glob import glob

import logging
from utils import utils
import pdb

from Bio.Entrez import efetch
from Bio import Entrez

import xml.etree.ElementTree as ET

Entrez.email = 'yano0828@gmail.com'


def get_abstract(pmid):

    try:
        handle = efetch(db='pubmed', id=pmid,
                        retmode='xml', rettype='abstract')
        buff = handle.read()
    except:
        print("exception efetch")
        return None 

    try:
        root = ET.fromstring(buff)    
    except:
        raise ValueError("error")

    elm = root.find(".//Journal")
    try:
        volume = elm.find(".//Volume").text
    except:
        volume = ""
    try:
        issue = elm.find(".//Issue").text
    except:
        issue = ""
    try:
        year = elm.find(".//Year").text
    except:
        year = ""
    try:
        month = elm.find(".//Month").text
    except:
        month = ""
    try:
        day = elm.find(".//Day").text
    except:
        day = ""
    page = root.find(".//Pagination")
    try:
        startpage = page.find(".//StartPage").text
    except:
        startpage = ""
    try:
        endpage = page.find(".//EndPage").text
    except:
        endpage = ""

    journal_title = elm.find(".//Title").text
    ios = elm.find(".//ISOAbbreviation").text
    print(f"{year}/{month}/{day}")
    print(f"{journal_title}/{ios}")
    print(f"{volume}/{issue}/{startpage}-{endpage}")

    elm = root.find(".//ArticleTitle")
    title = ET.tostring(elm, encoding='utf-8', method='text').decode('utf-8') 
    print(f"{title}")

    meta = {}
    
    meta["publication"]={"year":year, "month":month, "day":day}
    meta["journal"]={"title":journal_title, "ios":ios, "vol":volume, "issue":issue}
    meta["pagenation"]={"start":startpage, "end":endpage}
    meta["title"]=title

    return meta


def gen_AbstractJsonData(txt, ann, pmid):

    try:
        meta_info = get_abstract(pmid)
    except:
        raise ValueError("error!")
        

    json_data = {}

    with open(txt) as fp:
        txt = fp.read()

    with open(ann) as fp:
        lines = fp.readlines()
    lines = [line.strip() for line in lines]

    entity_list = []

    for i in range(0, len(lines), 2):
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

        entity_list.append(entity)

    for key, val in meta_info.items():
        json_data[key] = val 

    json_data['pmid'] = pmid
    json_data['text'] = txt
    json_data['entities'] = entity_list
    now = datetime.now()
    json_data['last_modified'] = now.strftime("%Y-%m-%dT%H:%M:%S")

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
    for i in range(0, len(lines), 2):
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

        entity_list.append(entity)

    json_data['entities'] = entity_list
    now = datetime.now()
    json_data['last_modified'] = now.strftime("%Y-%m-%dT%H:%M:%S")

    index_data = {}
    index_data['_index'] = "entity"
    index_data['_type'] = "cancer_immunology"
    index_data['_id'] = pmid

    return json_data, index_data


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

    extract_dir = parameters["pubmed_extract_dir"]
    annotate_dir = parameters["pubmed_annotate_dir"]

    txt_files = sorted(glob(f"{extract_dir}/*txt"))
    ann_files = sorted(glob(f"{annotate_dir}/*ann"))

    # abstract index

    json_dataset = []
    for txt, ann in zip(txt_files, ann_files):
        path, txt_fname = os.path.split(txt)
        path, ann_fname = os.path.split(ann)
        txt_basename, _ = os.path.splitext(txt_fname)
        ann_basename, _ = os.path.splitext(ann_fname)
        assert (txt_basename == ann_basename)
        print(txt, ann)
        try:
            json_data, index_data = gen_AbstractJsonData(txt, ann, ann_basename)
            json_dataset.append((json_data, index_data))
        except:
            pass

    output_dir = parameters["output_dir"]
    with open(os.path.join(output_dir, 'abstract.jsonl'), 'w') as fout:
        for json_data, index_data in json_dataset:
            json.dump({'index': index_data}, fout)
            fout.write('\n')
            json.dump(json_data, fout)
            fout.write('\n')

    # entity index

    json_dataset = []
    for txt, ann in zip(txt_files, ann_files):
        path, txt_fname = os.path.split(txt)
        path, ann_fname = os.path.split(ann)
        txt_basename, _ = os.path.splitext(txt_fname)
        ann_basename, _ = os.path.splitext(ann_fname)
        assert (txt_basename == ann_basename)
        print(txt, ann)
        json_data, index_data = gen_EntityJsonData(txt, ann, ann_basename)
        json_dataset.append((json_data, index_data))

    output_dir = parameters["output_dir"]
    with open(os.path.join(output_dir, 'entity.jsonl'), 'w') as fout:
        for json_data, index_data in json_dataset:
            json.dump({'index': index_data}, fout)
            fout.write('\n')
            json.dump(json_data, fout)
            fout.write('\n')

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))

if __name__ == '__main__':                                                                                                                        
    main()
