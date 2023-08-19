
import pdb
import xmltodict
from nltk.corpus import wordnet as wn
import sys
import os
import re
import itertools
import logging

import numpy as np
import math
import time
import pickle
from nltk.util import ngrams

from utils import utils
import nltk
nltk.download('wordnet')


def replace_greek_char(term):
    name = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda',
            'mu', 'nu', 'xi', 'omicron', 'pi', 'pho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']

    letter = list('αβγδεζηθικλμνξοπρστυφχψω')

    term_copy = term
    assert (len(name) == len(letter))

    capitalized_name = [n.capitalize() for n in name]

    tokens = []
    for token in term.split():
        for n, cap_n, l in zip(name, capitalized_name, letter):
            if token.startswith(n) or token.endswith(n):
                token = token.replace(n, l)
            if token.startswith(cap_n) or token.endswith(cap_n):
                token = token.replace(cap_n, l)
        tokens.append(token)

    term = ' '.join(tokens)

    return term


def replace_positive_negative(term):

    term = term.lower()
    if "positive" in term:
        term = term.replace("positive", "(+)")
    elif "negative" in term:
        term = term.replace("negative", "(-)")
    return term


def make_ngram(terms, ngram=4):

    ngram_terms = []
    for n in range(1, ngram):
        ngram_terms += list(ngrams(terms, n))

    return ngram_terms


def make_synonym_dict(umls_atoms):

    flatten_atom_names = umls_atoms["flatten_atom_names"]
    atom_synonyms = umls_atoms["atom_synonyms"]
    synonyms = list(set(tuple(l) for l in atom_synonyms))

    atom_synonyms = []
    for atoms in synonyms:
        atoms = sorted(list(set([re.sub('\(.+\)', '', atom).strip()
                       for atom in atoms if not re.match('\d+', atom)])))
        # synonyms for the atom
        atom_synonyms.append(atoms)

    # sorted all atoms
    all_atoms = sorted(
        list(set([atom for syms in atom_synonyms for atom in syms])))

    synonym_map = {}
    for atom in all_atoms:
        indexes = [k for k, syns in enumerate(atom_synonyms) if atom in syns]
        # map atom to the index to synonym list
        synonym_map[atom] = indexes

    return all_atoms, synonym_map, atom_synonyms


def expand_dict(dict_path, parameters):

    print(dict_path)

    with open(dict_path, 'rb') as fp:
        umls_atoms = pickle.load(fp)

    terms, synonym_map, term_synonyms = make_synonym_dict(umls_atoms)

    new_terms = terms.copy()
    for term in terms:

        # greek character replacement
        term_greek = replace_greek_char(term)
        if term_greek != term:
            new_terms.append(term_greek)
            synonym_map[term_greek] = synonym_map[term]

        # positive negative replacement
        term_posneg = replace_positive_negative(term)
        if term_posneg != term:
            new_terms.append(term_posneg)
            synonym_map[term_posneg] = synonym_map[term]

    new_terms = sorted(list(set(new_terms)))

    term_dict = {}
    term_dict['terms'] = new_terms
    term_dict['synonym_map'] = synonym_map
    term_dict['term_synonyms'] = term_synonyms

    return term_dict


def disambiguate_dict(dicts, parameters):

    ent_dep = parameters["entity_dependency"]

    for par_ent, cld_ents in ent_dep.items():

        par_terms = dicts[par_ent]
        cld_terms = []
        for cld_ent in cld_ents:
            cld_terms += dicts[cld_ent]

        new_terms = set(par_terms).difference(set(cld_terms))
        dicts[par_ent] = sorted(new_terms)

    return


def disambiguate_dict_old(dicts, parameters):

    tcell_key = 't-lymphocyte_dict_file'

    entity_types = list(dicts.keys())

    tcell_terms = dicts[tcell_key]
    entity_types.remove(tcell_key)

    print("Before disambiguation")
    for key, terms in dicts.items():
        print(key, len(terms))

    for etype in entity_types:

        # skip the disambiguation process for cell
        if etype == "cell_dict_file":
            continue

        terms = dicts[etype]
        new_terms = set(terms).difference(set(tcell_terms))
        dicts[etype] = sorted(new_terms)

    print("After disambiguation")
    for key, terms in dicts.items():
        print(key, len(terms))

    return


def save_dict(dicts, parameters):

    newdir = parameters['processed_dict_dir']
    utils.makedir(newdir)

    for key, items in dicts.items():

        terms = items['terms']
        synonym_map = items['synonym_map']
        term_synonyms = items['term_synonyms']

        fname = f'{key}.dict'
        new_dict_path = os.path.join(newdir, fname)

        with open(new_dict_path, 'w') as fp:
            for term in terms:
                fp.write('{}\n'.format(term))

        fname = f'{key}_synonyms.pkl'
        new_dict_path = os.path.join(newdir, fname)

        with open(new_dict_path, 'wb') as fp:
            pickle.dump(items, fp)


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
        parameters['dict_dir'], f'{etype}_dict.pkl') for etype in entity_types]

    new_dict = {}
    for etype, dict_path in zip(entity_types, dict_paths):
        term_dict = expand_dict(dict_path, parameters)
        new_dict[etype] = term_dict

    # disambiguate_dict(new_dict, parameters)

    save_dict(new_dict, parameters)

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()
