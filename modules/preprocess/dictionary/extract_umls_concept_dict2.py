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

import pdb

def prepare_umls_dict(parameters):

    umls_meta_dir = parameters["umls_meta_files_dir"]
    mrconso_rrf = os.path.join(umls_meta_dir, "MRCONSO.RRF")
    mrrel_rrf = os.path.join(umls_meta_dir, "MRREL.RRF")

   
    cui_dict = defaultdict(list)
    cui_rel_dict = defaultdict(list)

    with open(mrconso_rrf) as fp:
        for line in fp:
            fields = line.strip().split('|')
            cui = fields[0]
            lang = fields[1]
            atom = fields[-5]

            if lang == "ENG":
                cui_dict[cui].append(atom)
            
    cui_dict = {key: list(set(val)) for key, val in cui_dict.items()}

    with open(mrrel_rrf) as fp:
        for line in fp:
            fields = line.strip().split('|')
            cui = fields[0]
            rel_cui = fields[4]
            cui_rel_dict[cui].append(rel_cui)

    cui_rel_dict = {key: list(set(val)) for key, val in cui_rel_dict.items()}

    return cui_dict, cui_rel_dict


def generate_umls_dict(target_cui, dict_path, cui_dict, cui_rel_dict):

    umls_atoms = {}

    atoms = cui_dict[target_cui]

    for cui in cui_rel_dict[target_cui]:
        atoms += cui_dict[cui]

    atoms = sorted(list(set(atoms)))

    umls_atoms["all_synonyms"] = atoms 
    
    pdb.set_trace()
    with open(dict_path, 'bw') as fp:
        pickle.dump(umls_atoms, fp)


def main():

    # check running time                                                                                                   
    t_start = time.time()                                                                                                  
    UMLS_cui = {}
    UMLS_cui['cytokine'] = "C0079189"
    UMLS_cui['transcription_factor'] = "C0040648"
    UMLS_cui['t_lymphocyte'] = "C0039194"
    #UMLS_cui['disease'] = "C0012634"
    #UMLS_cui['chemicals'] = "C0220806"


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

    cui_dict, cui_rel_dict = prepare_umls_dict(parameters)

    utils.makedir(parameters['dict_dir'])
    for concept_name, cui in UMLS_cui.items():
        print("-"*20)
        print("generating UMLS concept dict",concept_name, cui)

        filename = f"{concept_name}_dict.pkl" 
        dict_path = os.path.join(parameters['dict_dir'], filename)
        generate_umls_dict(cui, dict_path, cui_dict, cui_rel_dict)


    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))

if __name__ == '__main__':                                                                                                                        
    main()







