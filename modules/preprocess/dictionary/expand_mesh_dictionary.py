
import sys
import os
import re
import itertools
import logging

import numpy as np
import math
import time
from nltk.util import ngrams

from utils import utils
import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn


import xmltodict

import pdb


def get_synonyms(word):
    """Get the synonyms of word from Wordnet."""
    lemmas = set().union(*[s.lemmas() for s in wn.synsets(word)])
    return list(set(l.name().lower().replace("_", " ") for l in lemmas) - {word})


def replace_greek_char(term):
    name = ['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa', 'lambda','mu','nu','xi','omicron','pi','pho','sigma','tau','upsilon','phi','chi','psi','omega']

    letter = list('αβγδεζηθικλμνξοπρστυφχψω')

    term_copy = term
    assert(len(name) == len(letter)) 

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


def make_ngram(terms, ngram = 4):

    ngram_terms = []
    for n in range(1, ngram):
        ngram_terms += list(ngrams(terms, n))

    return ngram_terms

def expand_dict(dict_path, parameters):

    print(dict_path)

    with open(dict_path) as fp:
        terms = [l.strip() for l in fp.readlines() if len(l) > 0]

    new_terms = terms.copy()
    for term in terms:
        subterms = re.split(r',? ', term)
        ngram_subterms = make_ngram(subterms, 4)

        # greek character replacement
        term_greek = replace_greek_char(term)
        if term_greek != term:
            subterms = re.split(r',? ', term_greek)
            ngram_subterms += make_ngram(subterms, 4)

        # positive negative replacement
        term_posneg = replace_positive_negative(term)
        if term_posneg != term:
            subterms = re.split(r',? ', term_posneg)
            ngram_subterms += make_ngram(subterms, 4)
    
        ngrams_subterms = list(set(ngram_subterms))

        for ngram in ngrams_subterms:
            new_terms.append(' '.join(ngram))

    # remove term if it contains only numbers
    terms_ = [term for term in new_terms if not re.match('\d+', term)]

    # remove term if it has any synonyms, which indicate the term is a word. 
    new_terms = [term for term in terms_ if len(get_synonyms(term)) == 0]


    newdir = parameters['processed_dict_dir']
    utils.makedir(newdir)
    _, fname = os.path.split(dict_path)
    new_dict_path = os.path.join(newdir, fname)
    

    new_terms = sorted(list(set(new_terms)))

    return new_terms

#    print(new_terms)
#    with open(new_dict_path, 'w') as fp:
#        for term in new_terms:
#            fp.write('{}\n'.format(term))
#                

def disambiguate_dict(dicts):
    
    tcell_key = 't-lymphocyte_dict_file'

    entity_types = list(dicts.keys())

    tcell_terms = dicts[tcell_key]
    entity_types.remove(tcell_key)

    print("Before disambiguation")
    for key, terms in dicts.items():
        print(key, len(terms))

    for etype in entity_types:
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

    for key, terms in dicts.items():

        fname = parameters[key]
        new_dict_path = os.path.join(newdir, fname)

        with open(new_dict_path, 'w') as fp:
            for term in terms:
                fp.write('{}\n'.format(term))


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
    #logger.addHandler(handler2)
                                                                                                                           
    # set config path by command line                                                                                      
    inp_args = utils._parsing()                                                                                            
    config_path = getattr(inp_args, 'yaml')                                                                                
    with open(config_path, 'r') as stream:                                                                                 
        parameters = utils._ordered_load(stream)

    # print config 
    utils._print_config(parameters, config_path)

    entity_types = ['cytokine_dict_file',
                    'tf_dict_file',
                    't-lymphocyte_dict_file',
                    'protein_dict_file']

    dict_paths = [os.path.join(parameters['dict_dir'],parameters[etype]) for etype in entity_types]

    new_dict = {}
    for etype, dict_path in zip(entity_types, dict_paths):
        new_terms = expand_dict(dict_path, parameters)
        new_dict[etype] = new_terms

    disambiguate_dict(new_dict)

    save_dict(new_dict, parameters)


    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))

if __name__ == '__main__':                                                                                                                        
    main()


