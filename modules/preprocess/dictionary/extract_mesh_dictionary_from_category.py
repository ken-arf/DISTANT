
import sys
import os
import itertools
import logging

import numpy as np
import math
import time

from utils import utils

import xmltodict

import pdb

def get_term_string(record):
    concept = record['ConceptList']['Concept']

    if type(concept) != list:
        termlist = concept['TermList']['Term']
        if type(termlist) == list:
            strings = [term['String'] for term in termlist]
        else:
            strings = [termlist['String']]
    else:
        strings = []
        for con in concept:
            termlist = con['TermList']['Term']
            if type(termlist) == list:
                strings += [term['String'] for term in termlist]
            else:
                strings += [termlist['String']]
    
    return strings

def is_decendant(treenumber, root_tree_number):

    nodes = treenumber.split('.')
    if root_tree_number == '.'.join(nodes[:-1]):
        return True
    else:
        return False

def extract_nrw_term(root_tree_number, doc):

    print("extract_nrw_term")
    print("root_tree_number", root_tree_number)
    records = doc['DescriptorRecordSet']['DescriptorRecord']

    child_terms = []
    for record in records:
        try:
            treenumberlist = record['TreeNumberList']['TreeNumber']
        except:
            continue

        if type(treenumberlist) != list:
            treenumberlist = [treenumberlist]

        for treenumber in treenumberlist:
            if is_decendant(treenumber, root_tree_number):
                print("found child", treenumber)
                yield get_term_string(record)
                yield from extract_nrw_term(treenumber, doc)

    
def extract_term(term, doc, dict_path):
    print(term)
    print(doc.keys())
    print(dict_path)

    all_terms = [[term]]

    records = doc['DescriptorRecordSet']['DescriptorRecord']
    print(len(records))
    for record in records:

        if (record['DescriptorName']['String'] == term):
            concept = record['ConceptList']['Concept']
      
            if type(concept) == list:
                concept = concept[0]
            treenumberlist = record['TreeNumberList']['TreeNumber']
            termlist = concept['TermList']['Term']
            strings = [term['String'] for term in termlist]
            break

    if type(treenumberlist) != list:
        treenumberlist = [treenumberlist]

    print('debug')
    print(dict_path)

    with open(dict_path, 'w') as fp:
        for treenumber in treenumberlist:
            for terms in extract_nrw_term(treenumber, doc):
                print(terms)
                for term in terms:
                    fp.write('{}\n'.format(term))


def extract_terms(record):
    
    terms = []
    if "ConceptList" in record:
        concepts = record['ConceptList']['Concept']
        if type(concepts) == dict:
            concepts = [concepts]
        for concept in concepts:
            try:
                termlist = concept['TermList']['Term']
                strings = [term['String'] for term in termlist]
                terms += strings
            except:
                pass
    return terms
            

def collect_category_terms(alphabet_suffix, doc):


    records = doc['DescriptorRecordSet']['DescriptorRecord']
    print(len(records))

    term_list = []
    for record in records:
        if 'TreeNumberList' in record:
            relevant = False
            treenumberlist = record['TreeNumberList']['TreeNumber']
            if type(treenumberlist) == str:
                treenumberlist = [treenumberlist]

            for number in treenumberlist:
                print(number)
                if number.startswith(alphabet_suffix):
                    relevant = True
                    break

            if relevant:
                terms = extract_terms(record)
                term_list += terms

    return term_list

def main():

    # check running time                                                                                                   
    t_start = time.time()                                                                                                  

    filenames = ["disease.dict", 
                "chemicals.dict"]

    mesh_terms = ["Disease", 
                "Chemicals"]

    mesh_category = ["C", 
                "D"]

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


    xml_path = os.path.join(parameters['corpus_dir'], parameters['corpus_file'])
    with open(xml_path) as fd:
        mesh_doc = xmltodict.parse(fd.read())

    utils.makedir(parameters['dict_dir'])

    for fname, category in zip(filenames, mesh_category):
        dict_path = os.path.join(parameters['dict_dir'], fname)
        term_list = collect_category_terms(category, mesh_doc)
        print(category, len(term_list))

        term_list = sorted(list(set(term_list)))
        with open(dict_path, 'w') as fp:
            for k, term in enumerate(term_list):
                print(term)
                fp.write(f'{term}\n')


    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))

if __name__ == '__main__':                                                                                                                        
    main()


