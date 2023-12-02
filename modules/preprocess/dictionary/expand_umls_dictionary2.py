
import pdb
from utils import utils
from nltk.corpus import stopwords
import sys
import os
import re
import itertools
import logging

import numpy as np
import math
import time
import pickle

import nltk
nltk.download('stopwords')


def unique_dict(dict_path, parameters):

    entries = []
    with open(dict_path) as fp:
        for line in fp:
            entries.append(line.strip())

    entries = list(set(entries))
    return entries


def expand_dict_old(dict_path, parameters, cui_dict, cui_rel_dict):

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


def expand_dict(dicts, parameters):

    max_term_len = 50

    stops = list(set(stopwords.words('english')))

    newdir = parameters['processed_dict_dir']
    utils.makedir(newdir)

    for etype, entries in dicts.items():

        fname = f'{etype}.dict'
        new_dict_path = os.path.join(newdir, fname)

        with open(new_dict_path, 'w') as fp:
            for entry in entries:
                fields = entry.strip().split('\t')
                term = fields[0]
                term_lc = fields[0].lower()
                ref = fields[1]

                if len(term) > max_term_len:
                    continue

                fp.write(f'{term}|{term_lc}|{ref}\n')

                if ',' in term:
                    first_phrase = term_lc.split(',')[0]
                    fp.write(f'{first_phrase}|{first_phrase.lower()}|{ref}\n')


def save_dict(dicts, parameters):

    stops = list(set(stopwords.words('english')))

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
                term_phrase = term_lc.split(',')[0]
                term_first_word = term_phrase.split(' ')[0]

                if term_first_word in stops:
                    continue

                fp.write(f'{term}|{term_lc}|{term_phrase}|{cui}\n')

                # changed 2023/10/14
                if term_phrase != term_first_word:
                    pattern = r"^[A-Za-z][A-Za-z1-9]+"
                    m = re.match(pattern, term_first_word)
                    if m:
                        fp.write(f'{term}|{term_lc}|{term_first_word}|{cui}\n')


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

    entity_types = parameters["entity_types"]

    dict_paths = [os.path.join(
        parameters['dict_dir'], f'{etype}_dict.txt') for etype in entity_types]

    new_dict = {}
    for etype, dict_path in zip(entity_types, dict_paths):
        term_dict = unique_dict(dict_path, parameters)
        new_dict[etype] = term_dict

    expand_dict(new_dict, parameters)

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()
