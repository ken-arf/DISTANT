
import os
import sys
import re
from collections import defaultdict
import random
import time
from tqdm import tqdm
from glob import glob
import numpy as np
import spacy


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import editdistance as ed

import spacy
from scispacy.abbreviation import AbbreviationDetector
import scispacy
from spacy.lang.en import English

import nltk
# nltk.download("wordnet", quiet=True)
from nltk import word_tokenize
# nltk.download('punkt')
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer, LancasterStemmer


from snorkel.augmentation import ApplyOnePolicy, PandasTFApplier
from snorkel.augmentation import transformation_function
from snorkel.labeling import PandasLFApplier, LFApplier, LFAnalysis, labeling_function
from snorkel.analysis import get_label_buckets
from snorkel.labeling.model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel

from snorkel.labeling.apply.spark import SparkLFApplier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from utils import utils
from entity_extract import ExtractEntityCandidate

import itertools

import pdb


ABSTAIN = -1

# cancer immunology
CYTOKINE = 0
TRANSCRIPTION_FACTOR = 1
T_LYMPHOCYTE = 2

# jnlpba
PROTEIN = 0
CELL_LINE = 1
CELL = 2
DNA = 3
RNA = 4

# BC5CDR
CHEMICAL = 0
DISEASE = 1


# threshold for edit distance
max_dist = 0
# max_dist=1

# nlp = spacy.load("en_core_sci_lg")
nlp = spacy.load("en_core_sci_sm")

# nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("sentencizer")

lemmatizer = WordNetLemmatizer()

greek_name = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron',
              'pi', 'pho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']
greek_letter = list('αβγδεζηθικλμνξοπρστυφχψω')

greek_translate = {l: w for l, w in zip(greek_letter, greek_name)}

synonym_table = {
    'lymphocyte': 'cell',
    'lymphocyte': 'lymph cell',
    'cell': 'lymphocyte',
    '+': 'positive',
    '-': 'negative',
}

synonym_table.update(greek_translate)

lemmatizer = WordNetLemmatizer()


def is_substr(ref, src):

    ref = re.sub(r'\W', '', ref)
    src = re.sub(r'\W', '', src)

    if src in ref:
        return True
    else:
        return False



def min_edit_distance(ref, src):

    min_l = ed.eval(ref, src)

    if min_l > 0:
        if src.startswith(ref):
            try:
                pattern = f"{ref}[αβγA-Za-z1-9\+\-\_\(\)]+"
                m = re.match(pattern, src)
                if m:
                    if m.group(0) == src:
                        min_l = 0
            except:
                pass

    if min_l > 0:
        ref = re.sub(r'\W', '', ref)
        src = re.sub(r'\W', '', src)
        if src == ref:
            min_l = 0

    return min_l
        


def entity_extract(entityExtraction, sent, pmid, k):

    entities = []
    start_tokens = []
    end_tokens = []
    start_chars = []
    end_chars = []

    candidates = entityExtraction.extract_candiate(
        sent, custom_model=True, scipy_model=False)

    for ent in candidates:

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
                       'text': [sent] * len(entities),
                       'pmid': [f"{pmid}_{k}"] * len(entities)})

    return df


# snorkel Labeling functions
@labeling_function()
def lf_cytokine_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['cytokine.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CYTOKINE
    return ABSTAIN


@labeling_function()
def lf_cytokine_substr(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['cytokine.dict']:
        if is_substr(phrase, ent):
            return CYTOKINE
    return ABSTAIN


@labeling_function()
def lf_transcription_factor_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['transcription_factor.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return TRANSCRIPTION_FACTOR
    return ABSTAIN


@labeling_function()
def lf_transcription_factor_substr(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['transcription_factor.dict']:
        if is_substr(phrase, ent):
            return TRANSCRIPTION_FACTOR
    return ABSTAIN


@labeling_function()
def lf_t_lymphocyte_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['t_lymphocyte.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return T_LYMPHOCYTE
    return ABSTAIN


@labeling_function()
def lf_t_lymphocyte_substr(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['t_lymphocyte.dict']:
        if is_substr(phrase, ent):
            return T_LYMPHOCYTE
    return ABSTAIN


@labeling_function()
def lf_protein_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['protein.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return PROTEIN
    return ABSTAIN


@labeling_function()
def lf_cell_line_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['cell_line.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CELL_LINE
    return ABSTAIN


@labeling_function()
def lf_cell_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['cell.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CELL
    return ABSTAIN


@labeling_function()
def lf_dna_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['dna.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return DNA
    return ABSTAIN


@labeling_function()
def lf_rna_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['rna.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return RNA
    return ABSTAIN


@labeling_function()
def lf_debug(x):
    ent = x.lower()
    for phrase in dist_dict['rna.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            # l = ed.eval(ent, phrase)
            # if l == 0:
            return RNA
    return ABSTAIN


@labeling_function()
def lf_chemicals_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['chemicals.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CHEMICAL
    return ABSTAIN


@labeling_function()
def lf_chemicals_substr(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['chemicals.dict']:
        if is_substr(phrase, ent):
            return CHEMICAL
    return ABSTAIN


@labeling_function()
def lf_disease_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['disease.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return DISEASE
    return ABSTAIN


@labeling_function()
def lf_disease_substr(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['disease.dict']:
        if is_substr(phrase, ent):
            return DISEASE
    return ABSTAIN


global dist_dict


def snorkel(parameters, df_train, lfs):

    global dist_dict

    lf_applier = PandasLFApplier(lfs=lfs)
    L_train = lf_applier.apply(df=df_train)

    return L_train


def snorkel_spark(parameters, df_train, lfs):

    global dist_dict

    sc = SparkContext()
    rdd = sc.parallelize(df_train)
    lf_applier = SparkLFApplier(lfs=lfs)
    L_train = lf_applier.apply(rdd)

    return L_train


def main():

    global dist_dict

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

    entityExtraction = ExtractEntityCandidate(
        parameters["segmentation_predict_config"])

    dfs = []
    for document_path in tqdm(sorted(documents)):
        _, fname = os.path.split(document_path)
        pmid, _ = os.path.splitext(fname)
        with open(document_path) as fp:
            text = fp.read().strip()
            doc = nlp(text)
            for k, sent in enumerate(doc.sents):
                # df=entity_extract(nlp(sent.text), pmid,k)

                try:
                    df = entity_extract(entityExtraction, sent.text, pmid, k)
                    dfs.append(df)
                except:
                    print("Exception, entity_extract")
                    

    df_train = pd.concat(dfs, ignore_index=True)

    print("df_train size:", df_train.shape)

    # dictionary generation1
    dict_dirs = parameters["processed_dict_dirs"]
    dictionary_files = parameters["dict_files"]

    dist_dict = defaultdict(list)
    for fname in dictionary_files:

        terms = []
        for dict_dir in dict_dirs:
            path = os.path.join(dict_dir, fname)
            with open(path) as fp:
                lines = [l.strip() for l in fp.readlines()]
            entries = sum([line.split('|')[1:3] for line in lines], [])
            terms += entries

        dist_dict[fname] = sorted(list(set(terms)))

    # snorkel labeling functions
    dictionary_files = parameters["dict_files"]
    lfs = []
    for fname in dictionary_files:
        basename, _ = os.path.splitext(fname)
        lf_func = f"lf_{basename}_distsv"
        lfs.append(eval(lf_func))
        # lf_func = f"lf_{basename}_substr"
        # lfs.append(eval(lf_func))

    if parameters["spark"]:
        L_train = snorkel_spark(parameters, df_train["entities"], lfs)
    else:
        L_train = snorkel(parameters, df_train["entities"], lfs)

    # snorkel result summary
    print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

    # load snorkel labeling results
    for i in range(len(lfs)):
        df_train[lfs[i].name] = L_train[:, i]

    # label_model = MajorityLabelVoter(cardinality=int(len(lfs) / 2))
    label_model = MajorityLabelVoter(cardinality=len(lfs))
    df_train["label"] = label_model.predict(L=L_train)

    df_train['start_tokens'] = df_train['start_tokens'].astype(np.int64)
    df_train['end_tokens'] = df_train['end_tokens'].astype(np.int64)
    df_train['start_chars'] = df_train['start_chars'].astype(np.int64)
    df_train['end_chars'] = df_train['end_chars'].astype(np.int64)

    # drop na rows
    df_train = df_train.dropna()

    df_train_raw = df_train.copy()

    N = df_train_raw.shape[0]
    labels = sorted(list(df_train_raw["label"].unique()))
    for l in labels:
        n = df_train_raw[df_train_raw["label"] == l].shape[0]
        print(f"label: {l}: {n}/{N}")

    # filter negative samples
    df_train = df_train[df_train.label != ABSTAIN]

    N = df_train.shape[0]
    labels = sorted(list(df_train["label"].unique()))
    for l in labels:
        n = df_train[df_train["label"] == l].shape[0]
        ratio = float(n)/N
        print(f"label: {l}: {n}/{N} ({ratio:.2f})")

    corpus_dir = parameters["corpus_dir"]
    utils.makedir(corpus_dir)

    df_train_raw.to_csv(os.path.join(corpus_dir, "df_train_pos_neg.csv"))
    df_train.to_csv(os.path.join(corpus_dir, "df_train_pos.csv"))
    df_train_core = df_train[["entities", "text", "pmid", "label"]]
    df_train_core.to_csv(os.path.join(corpus_dir, "df_train_pos_clean.csv"))

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()
