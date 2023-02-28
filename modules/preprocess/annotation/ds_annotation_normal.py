
import os
import sys
import re
import spacy
import random
import numpy as np
import time

from tqdm import tqdm

from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import editdistance as ed

import spacy
from scispacy.abbreviation import AbbreviationDetector
import scispacy
from spacy.lang.en import English

import nltk
nltk.download("wordnet", quiet=True)
from nltk import word_tokenize
nltk.download('punkt')
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer, LancasterStemmer


from snorkel.augmentation import ApplyOnePolicy, PandasTFApplier
from snorkel.augmentation import transformation_function
from snorkel.labeling import PandasLFApplier, LFApplier, LFAnalysis, labeling_function
from snorkel.analysis import get_label_buckets
from snorkel.labeling.model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from utils import utils

import itertools

import pdb


ABSTAIN = -1

# cancer immunology
CYTOKINE=0
TRANSCRIPTION_FACTOR=1
T_LYMPHOCYTE=2

# jnlpba
PROTEIN=0
CELL_LINE=1
CELL=2  
DNA=3
RNA=4


# threshold for edit distance
max_dist=0

#nlp = spacy.load("en_core_sci_lg")
nlp = spacy.load("en_core_sci_sm")

nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("sentencizer")

lemmatizer = WordNetLemmatizer()

greek_name = ['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa', 'lambda','mu','nu','xi','omicron',
              'pi','pho','sigma','tau','upsilon','phi','chi','psi','omega']
greek_letter = list('αβγδεζηθικλμνξοπρστυφχψω')
    
greek_translate = {l:w for l, w in zip(greek_letter, greek_name)}
    
synonym_table = {        
        'lymphocyte': 'cell',
        'lymphocyte': 'lymph cell',
        'cell': 'lymphocyte',
        '+':'positive',
        '-':'negative',    
}

synonym_table.update(greek_translate)

def min_edit_distance(ref, src):

    lemmatizer = WordNetLemmatizer()
    ref = ' '.join([lemmatizer.lemmatize(w.lower()) for w in ref.split()])
    src = ' '.join([lemmatizer.lemmatize(w.lower()) for w in src.split()])
    
    min_l = ed.eval(ref, src)
 
    # synonym word exchange
    for k, v in synonym_table.items():
        if not k in src:
            continue
       
        src2 = src.replace(k, v)
        l = ed.eval(ref, src2)
        if l < min_l:
                min_l = l
                src = src2

    if min_l > 0:
        ref = re.sub(r'\W', '', ref)
        src = re.sub(r'\W', '', src)
        if src == ref:
            min_l = 0

    return min_l



def entity_extract(sent, pmid,k):
       
   entities = []
   start_tokens = []
   end_tokens = []
   start_chars= []
   end_chars = []

   for ent in sent.ents:

       entities.append(ent.text)
       start_tokens.append(int(ent.start))
       end_tokens.append(int(ent.end))
       start_chars.append(int(ent.start_char))
       end_chars.append(int(ent.end_char))

   df = pd.DataFrame({'entities': entities, 
                              'start_tokens': start_tokens, 
                              'end_tokens': end_tokens, 
                              'start_chars': start_chars, 
                              'end_chars': end_chars,
                             'text': [sent.text] * len(entities),
                             'pmid': [f"{pmid}_{k}"] * len(entities)})
   return df
   


# snorkel Labeling functions
@labeling_function()
def lf_cytokine_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['cytokine.dict']:
        #if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CYTOKINE
    return ABSTAIN


@labeling_function()
def lf_transcription_factor_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['transcription_factor.dict']:
        #if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return TRANSCRIPTION_FACTOR
    return ABSTAIN


@labeling_function()
def lf_t_lymphocyte_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['t_lymphocyte.dict']:
        #if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return T_LYMPHOCYTE
    return ABSTAIN


@labeling_function()
def lf_protein_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['protein.dict']:
        #if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return PROTEIN
    return ABSTAIN


@labeling_function()
def lf_cell_line_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['cell_line.dict']:
        #if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CELL_LINE 
    return ABSTAIN


@labeling_function()
def lf_cell_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['cell.dict']:
        #if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CELL 
    return ABSTAIN


@labeling_function()
def lf_dna_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['dna.dict']:
        #if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return DNA 
    return ABSTAIN


@labeling_function()
def lf_rna_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['rna.dict']:
        #if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return RNA 
    return ABSTAIN


def main():

    # check running time                                                                                                   
    t_start = time.time()                                                                                                  
    # set config path by command line                                                                                      
    inp_args = utils._parsing()                                                                                            
    config_path = getattr(inp_args, 'yaml')                                                                                
    with open(config_path, 'r') as stream:                                                                                 
        parameters = utils._ordered_load(stream)

    # print config                                                                                                         
    utils._print_config(parameters, config_path)


    # extract all entities by using spacy
    documents = glob(parameters["document_root_dir"] + "/*.txt")

    dfs = []
    for document_path in tqdm(sorted(documents)):
        _, fname = os.path.split(document_path)
        pmid, _ = os.path.splitext(fname)
        with open(document_path) as fp:
            text = fp.read().strip()
            doc = nlp(text)
            for k, sent in enumerate(doc.sents):
                df=entity_extract(nlp(sent.text), pmid,k)
                dfs.append(df)


    df_train = pd.concat(dfs, ignore_index=True)
    #df_train.head(100)


    # construction of the distant supervision dictionaries 
    Mesh_dict_dir= parameters["processed_dict_dir"]

    dictionary_files = parameters["dict_files"]


    global dist_dict
    dist_dict = {}
    for fname in dictionary_files:
        path = os.path.join(Mesh_dict_dir, fname)
        with open(path) as fp:
            lines = [l.strip() for l in fp.readlines()]
            lines_lower = [l.strip().lower() for l in fp.readlines()]
            dist_dict[fname] = lines + lines_lower 


    # snorkel labeling functions
    lfs = []
    for fname in dictionary_files:
        basename, _ = os.path.splitext(fname)
        lf_func = f"lf_{basename}_distsv"
        print(lf_func)
        lfs.append(eval(lf_func))

    #lfs = [lf_cytokine_distsv,lf_tf_distsv, lf_t_lymphocyte_distsv]


    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)
    LFAnalysis(L=L_train, lfs=lfs).lf_summary()

    # load snorkel labeling results
    for i in range(len(lfs)):
        df_train[lfs[i].name] = L_train[:, i]

    #df_train['lf_cytokine_distsv'] = L_train[:,0]
    #df_train['lf_tf_distsv'] = L_train[:,1]
    #df_train['lf_t_lymphocyte_distsv'] = L_train[:,2]


    label_model = MajorityLabelVoter(cardinality=len(lfs))
    df_train["label"] = label_model.predict(L=L_train)

    df_train['start_tokens'] = df_train['start_tokens'].astype(np.int64)
    df_train['end_tokens'] = df_train['end_tokens'].astype(np.int64)
    df_train['start_chars'] = df_train['start_chars'].astype(np.int64)
    df_train['end_chars'] = df_train['end_chars'].astype(np.int64)

    df_train_raw = df_train.copy()

    # filter negative samples
    df_train = df_train[df_train.label != ABSTAIN]

    
    corpus_dir = parameters["corpus_dir"]
    utils.makedir(corpus_dir)

    df_train_raw.to_csv(os.path.join(corpus_dir, "df_train_pos_neg.csv"))
    df_train.to_csv(os.path.join(corpus_dir, "df_train_pos.csv"))
    df_train_core=df_train[["entities","text","pmid","label"]]
    df_train_core.to_csv(os.path.join(corpus_dir, "df_train_pos_clean.csv"))

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':                                                                                                                        
    main()




