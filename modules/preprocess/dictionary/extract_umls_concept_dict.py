#!/usr/bin/env python
# coding: utf-8


import os
import sys
import requests
import argparse
import time
import re
from collections import defaultdict
import pickle

import logging

from utils import utils

uri= "https://uts-ws.nlm.nih.gov/rest/content/current" 
apiKey="3ebafc45-4529-4619-92a3-e3c9432b9e03"



def get_root_concept_atoms(cui):

    content_endpoint =  f"/CUI/{cui}/atoms"
    query={"apiKey": apiKey, 'language': 'ENG'}

    r = requests.get(uri+content_endpoint, params=query)
    r.encoding = "utf-8"


    if r.status_code != 200:
        print("request error")


    items = r.json()
    jsonData = items["result"]


    atom_names = [jdata["name"] for jdata in jsonData]


    urls = []
    for data in jsonData:
        print(data["name"])
        print(data["sourceConcept"])
        print(data["sourceDescriptor"])
        if data["sourceConcept"] != "NONE":
            urls.append(data["sourceConcept"])
        if data["sourceDescriptor"] != "NONE":
            urls.append(data["sourceDescriptor"])

    return atom_names, urls


def get_source_descendants(url):

    query={"apiKey": apiKey}
    content_endpoint = "/descendants"
    try:
        r = requests.get(url+content_endpoint, params=query,timeout=3.0)
        r.encoding = "utf-8"
        if r.status_code != 200:
            print("request error")
        items = r.json()
        jsonData = items["result"]
    except:
        jsonData = None
    return jsonData
    


def get_source_atoms(source_url):
    query={"apiKey": apiKey, "language": "ENG"}
    try:
        r = requests.get(source_url, params=query,timeout=3.0)
        r.encoding = "utf-8"
        if r.status_code != 200:
            print("request error")
        items = r.json()
        names = [item["name"] for item in items["result"]]
    except:
        names = None
    return names
    

def get_descend(url, all_names, url_descends_next):
    url = url.replace("/descendants","")
    
    jsonData = get_source_descendants(url)
    time.sleep(1.0)
    if jsonData == None:
        return
        
    for jdata in jsonData:
        atoms_url = jdata["atoms"]
        if jdata["descendants"] and jdata["descendants"] != "NONE":
            if not "SNOMED" in url:
                url_descends_next.append(jdata["descendants"])
        names = get_source_atoms(atoms_url)
        if names == None:
            continue
            
        time.sleep(1.0)
        print(names)
        all_names.append(names)


def get_recursive_descendants(all_names, url_descends):

    loop_cnt = 0
    while True:
        loop_cnt += 1
        print("get_recursive_descendants, loop", loop_cnt)
        url_descends_next = []

        n = len(url_descends)
        for k, url in enumerate(url_descends):
            print(f"loop_cnt:{loop_cnt}, {k}/{n}:{url}")
            get_descend(url, all_names, url_descends_next)
        
        if len(url_descends_next) == 0:
            break
        url_descends = url_descends_next.copy()


def generate_umls_dict(cui, dict_path):

    concept_atoms, urls = get_root_concept_atoms(cui)

    # step.1 get initial atoms for the concept
    all_names = []
    url_descends = []
    for url in urls:
        jsonData = get_source_descendants(url)
        time.sleep(1.0)
        if jsonData == None:
            continue
            
        for jdata in jsonData:
            atoms_url = jdata["atoms"]
            if jdata["descendants"] and jdata["descendants"] != "NONE":
                if not "SNOMED" in url:
                    url_descends.append(jdata["descendants"])
            names = get_source_atoms(atoms_url)
            if names == None:
                continue
                
            time.sleep(1.0)
            print(names)
            all_names.append(names)

    # step2. recursively extract descendants atoms
    get_recursive_descendants(all_names, url_descends)


    # insert root concept atoms at position 0
    all_names.insert(0, concept_atoms)
    sorted_names = sorted(list(set([name for names in all_names for name in names ])))
    # delete duplicate atoms
    all_names_clean = [sorted(list(set(names)))  for names in all_names]
    # flatten atoms names
    all_names_unfiltered = list(set([name for names in all_names_clean for name in names]))

    # step3. post-process atoms, which contain "," in the definition by splitting it into independent words 
    all_names_clean2 = []
    for names in all_names_clean:
        names_new = []
        for name in names:
            if ',' in name:
                words = [ w.strip() for w in name.split(",") ]
                for word in words:
                    if not word in names:
                        names_new.append(word)
                joint_words = ' '.join(words[::-1])
                if not joint_words in names:
                    names_new.append(joint_words)
            else:
                names_new.append(name)
            
        all_names_clean2.append(sorted(list(set(names_new))))
                    


    # step4. post-process the atoms generated at step.2 by deleting newly generated atoms which consists of alphabet and are not
    # the member of the atoms generated at step.1
    prog = re.compile(r'^[A-Za-z]+$')
    all_names_clean3 = []
    for names in all_names_clean2:
        # アルファベットだけの単語、かつ　フィルターリング処理前の　nameリストに含まれていなかった　単語を除いて登録する。
        new_names = [name for name in names if  not (re.match(prog, name) and (not name in all_names_unfiltered)) ]
        all_names_clean3.append(new_names)

    # save the result
    umls_atoms = {}
    flatten_atom_names = sorted(list(set([name for names in all_names_clean3 for name in names])))
    umls_atoms["flatten_atom_names"] = flatten_atom_names
    umls_atoms["atom_synonyms"] = all_names_clean3 
    
    with open(dict_path, 'bw') as fp:
        pickle.dump(umls_atoms, fp)


def main():

    # check running time                                                                                                   
    t_start = time.time()                                                                                                  
    UMLS_cui = {}
    UMLS_cui['cytokine'] = "C0079189"
    UMLS_cui['transcription_factor'] = "C0040648"
    UMLS_cui['t_lymphocyte'] = "C0039194"
    UMLS_cui['immune_checkpoint_inhibitors'] = "C4684977"
    UMLS_cui['protein'] = "C0033684"
    UMLS_cui['cell'] = "C0007634" 
    UMLS_cui['cell_line'] = "C0007600"
    UMLS_cui['rna'] = "C0035668"
    UMLS_cui['dna'] = "C0012854"


    t_start = time.time()

    # logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

    #handler2 = logging.FileHandler(filename="test.log")
    #handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(handler1)
    #logger.addHandler(handler2)
                                                                                                                           
    # set config path by command line
    inp_args = utils._parsing()                                                                                            
    config_path = getattr(inp_args, 'yaml')                                                                                
    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    # print config
    utils._print_config(parameters, config_path)

    utils.makedir(parameters['dict_dir'])
    for concept_name, cui in UMLS_cui.items():
        print("-"*20)
        print("generating UMLS concept dict",concept_name, cui)

        filename = f"{concept_name}_dict.pkl" 
        dict_path = os.path.join(parameters['dict_dir'], filename)
        generate_umls_dict(cui, dict_path)


    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))

if __name__ == '__main__':                                                                                                                        
    main()







