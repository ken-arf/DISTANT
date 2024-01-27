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

    cui_dict = {key: sorted(list(set(val)), key=lambda x: len(x))
                for key, val in cui_dict.items()}

    with open(mrrel_rrf) as fp:
        for line in fp:
            fields = line.strip().split('|')
            cui = fields[0]
            rel_cui = fields[4]
            cui_rel_dict[cui].append(rel_cui)

    cui_rel_dict = {key: list(set(val)) for key, val in cui_rel_dict.items()}

    return cui_dict, cui_rel_dict

def generate_umls_dict_(target_cui, cui_dict, cui_rel_dict, entries):

    atoms = cui_dict[target_cui][:]
    for atom in atoms:
        entries.append(f'{atom}\tUMLS:{target_cui}')

    next_cuis = []

    if target_cui in cui_rel_dict:
        for cui in cui_rel_dict[target_cui]:
            if not cui in cui_dict:
                continue

            atoms = cui_dict[cui][:]
            for atom in atoms:
                entries.append(f'{atom}\tUMLS:{cui}')
            next_cuis.append(cui)

    return list(set(next_cuis))


def generate_dict(root_cui, dict_path, cui_dict, cui_rel_dict, max_hop = 1):


    entries = []
    next_cuis = [root_cui]

    for hop in range(max_hop):

        narrower_cuis = []
        for target_cui in next_cuis:
            cuis = generate_umls_dict_(target_cui, cui_dict, cui_rel_dict, entries)
            narrower_cuis.extend(cuis)

        next_cuis = list(set(narrower_cuis))

    entries = list(set(entries))

    print(f'CUI {root_cui}: num of entries: {len(entries)}')

    with open(dict_path, 'w') as fp:
        for entry in entries:
            fp.write(f'{entry}\n')


def generate_umls_dict_old(target_cui, dict_path, cui_dict, cui_rel_dict):

    entries = []

    atoms = cui_dict[target_cui][:]
    for atom in atoms:
        entries.append(f'{atom}\tUMLS:{target_cui}')

    for cui in cui_rel_dict[target_cui]:
        atoms = cui_dict[cui][:]
        for atom in atoms:
            entries.append(f'{atom}\tUMLS:{cui}')

    with open(dict_path, 'w') as fp:
        for entry in entries:
            fp.write(f'{entry}\n')



def main():

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
    # logger.addHandler(handler2)

    # set config path by command line
    inp_args = utils._parsing()
    config_path = getattr(inp_args, 'yaml')
    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    # print config
    utils._print_config(parameters, config_path)



    UMLS_cui = {}
    #UMLS_cui['cytokine'] = "C0079189"
    #UMLS_cui['transcription_factor'] = "C0040648"
    #UMLS_cui['t_lymphocyte'] = "C0039194"
    #UMLS_cui['disease'] = "C0012634"
    #UMLS_cui['chemicals'] = "C0220806"

    for entity in parameters['entities'].keys():
        ri = parameters['entities'][entity]['RI']
        hops = parameters['entities'][entity]['UMLS_hops']
        UMLS_cui[entity] = {'RI':ri, 'UMLS_hops': hops}

    print(UMLS_cui)

    cui_dict, cui_rel_dict = prepare_umls_dict(parameters)

    utils.makedir(parameters['dict_dir'])
    for concept_name, mydict in UMLS_cui.items():
        print("-"*20)

        cui = mydict['RI']
        max_hops = mydict['UMLS_hops']

        print("generating UMLS concept dict", concept_name, cui)

        filename = f"{concept_name}_dict.txt"
        dict_path = os.path.join(parameters['dict_dir'], filename)

        generate_dict(cui, dict_path, cui_dict, cui_rel_dict, max_hop = max_hops)

    # save cui_dict, cui_rel_dict
    filename = f"cui_dict.pkl"
    dict_path = os.path.join(parameters['dict_dir'], filename)
    with open(dict_path, 'bw') as fp:
        pickle.dump(cui_dict, fp)

    filename = f"cui_rel_dict.pkl"
    dict_path = os.path.join(parameters['dict_dir'], filename)
    with open(dict_path, 'bw') as fp:
        pickle.dump(cui_rel_dict, fp)

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()
