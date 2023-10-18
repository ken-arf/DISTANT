
import sys
import os
import re
import itertools
import logging

import numpy as np
import math
import time
import pickle

from utils import utils


import pdb


def expand_dict(dict_path, parameters, cui_dict, cui_rel_dict):

    print(dict_path)

    with open(dict_path, 'rb') as fp:
        umls_atoms = pickle.load(fp)

    synonyms = umls_atoms['all_synonyms']

    term_dict = []
    for term in synonyms:
        term_lc = term.lower()
        for cui, vals in cui_dict.items():
            if term in vals:
                break

        term_dict.append((term, term_lc, cui))

    return term_dict


def save_dict(dicts, parameters):

    newdir = parameters['processed_dict_dir']
    utils.makedir(newdir)

    for etype, dict_ in dicts.items():

        fname = f'{etype}.dict'
        new_dict_path = os.path.join(newdir, fname)

        with open(new_dict_path, 'w') as fp:
            for entry in dict_:
                term = entry[0]
                term_lc = entry[1]
                cui = entry[2]
                term_head = term_lc.split(',')[0]
                fp.write(f'{term}|{term_lc}|{term_head}|{cui}\n')

                # changed 2023/10/14
                term_first = term_head.split(' ')[0]
                if term_head != term_first:
                    pattern = r"^[A-Z][A-Z1-9]+"
                    m = re.match(pattern, term_first)
                    if m:
                        fp.write(f'{term}|{term_lc}|{term_first}|{cui}\n')
                        


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

    with open(os.path.join(parameters['dict_dir'], 'cui_dict.pkl'), 'rb') as fp:
        cui_dict = pickle.load(fp)

    with open(os.path.join(parameters['dict_dir'], 'cui_rel_dict.pkl'), 'rb') as fp:
        cui_rel_dict = pickle.load(fp)

    entity_types = parameters["entity_types"]

    dict_paths = [os.path.join(
        parameters['dict_dir'], f'{etype}_dict.pkl') for etype in entity_types]

    new_dict = {}
    for etype, dict_path in zip(entity_types, dict_paths):
        term_dict = expand_dict(dict_path, parameters, cui_dict, cui_rel_dict)
        new_dict[etype] = term_dict

    save_dict(new_dict, parameters)

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()
