
import os
import sys
import importlib
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

from preprocess.annotation.entity_item import EntityItem

import itertools


#nlp = spacy.load("en_core_sci_scibert")
#nlp = spacy.load("en_ner_jnlpba_md")
nlp = spacy.load("en_core_sci_sm")

nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("sentencizer")


def get_synonyms(word):
    """Get the synonyms of word from Wordnet."""
    return wn.synsets(word.lower())

def entity_extract(sent, pmid,k):
       
   entities = []
   start_tokens = []
   end_tokens = []
   start_chars= []
   end_chars = []

   for ent in sent.ents:

       if len(get_synonyms(ent.text)) != 0:
            continue

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

    task = parameters["task_name"]
    entity_module = importlib.import_module(f"preprocess.annotation.defines.{task}.entity_define")


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

    entity_defines = entity_module.get_entity_defines(parameters)

    lfs = [entity_def.make_func() for entity_def in entity_defines]


    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)


    LFAnalysis(L=L_train, lfs=lfs).lf_summary()

    df_train['lf_cytokine_distsv'] = L_train[:,0]
    df_train['lf_tf_distsv'] = L_train[:,1]
    df_train['lf_t_lymphocyte_distsv'] = L_train[:,2]


    label_model = MajorityLabelVoter(cardinality=3)
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




