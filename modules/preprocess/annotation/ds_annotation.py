
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


CYTOKINE=0
TRANSCRIPTION_FACTOR=1
T_LYMPHOCYTE=2
PROTEIN=3
ABSTAIN  = -1

# threshold for edit distance
max_dist=0

#nlp = spacy.load("en_core_sci_scibert")
#nlp = spacy.load("en_ner_jnlpba_md")
nlp = spacy.load("en_core_sci_sm")

nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("sentencizer")


def get_synonyms(word):
    """Get the synonyms of word from Wordnet."""
    lemmas = set().union(*[s.lemmas() for s in wn.synsets(word)])
    return list(set(l.name().lower().replace("_", " ") for l in lemmas) - {word})

def entity_extract(sent, pmid,k):
       
   entities = []
   start_tokens = []
   end_tokens = []
   start_chars= []
   end_chars = []

   for ent in sent.ents:

#       if re.match(r'[a-z]+', ent.text):
#
#           continue
#   
#       if re.match(r'-?\d+', ent.text):
#           continue
           
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
   


def edit_distance(word1, word2):
    return ed.eval(word1, word2)


# snorkel Labeling functions
@labeling_function()
def lf_cytokine_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['cytokine.dict']:
        if ed.eval(ent,phrase.lower()) <= max_dist:
            return CYTOKINE
    return ABSTAIN



@labeling_function()
def lf_tf_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['tf.dict']:
        if ed.eval(ent,phrase.lower()) <= max_dist:
            return TRANSCRIPTION_FACTOR
    return ABSTAIN



@labeling_function()
def lf_t_lymphocyte_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['t-lymphocyte.dict']:
        if ed.eval(ent,phrase.lower()) <= max_dist:
            return T_LYMPHOCYTE
    return ABSTAIN



@labeling_function()
def lf_protein_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.entities.lower()
    for phrase in dist_dict['protein.dict'].split():
        if ed.eval(ent,phrase.lower()) <= max_dist:
            return PROTEIN
    return ABSTAIN




@transformation_function()
def tf_replace_word_with_synonym(x):
    """Try to replace a random word with a synonym."""
    words = x.text.lower().split()
    idx = random.choice(range(len(words)))
    synonyms = get_synonyms(words[idx])
    if len(synonyms) > 0:
        x.text = " ".join(words[:idx] + [synonyms[0]] + words[idx + 1 :])
        return x




@transformation_function()
def tf_replace_word_with_synonym_new(x):
    """Try to replace a random word with a synonym."""
    
    orig_entity = x.text[x.start_chars:x.end_chars]        
    #words = x.text.lower().split()
    words = [token.text.lower() for token in nlp(x.text)]
    
    while True:
        idx = random.choice(range(len(words)))
        if idx in range(x.start_tokens, x.end_tokens):
            continue
        else:
            break

    synonyms = get_synonyms(words[idx])
    if len(synonyms) > 0:
        x.text = " ".join(words[:idx] + [synonyms[0]] + words[idx + 1 :])
   
        try:
            x.start_chars = x.text.index(orig_entity.lower())
        except:
            print("not found")
            print(orig_entity.lower())
            print(x.text)
            
        # fix char position for entity
        x.end_chars = x.start_chars + len(orig_entity)
        x.entities = x.text[x.start_chars:x.end_chars]
        
        # fix token position for entity
        sym_len = len(synonyms[0].split())
        if x.end_tokens < idx:
            pass
        elif idx < x.start_tokens:
            x.start_tokens += sym_len - 1
            x.end_tokens += sym_len - 1
        
        x.pmid = x.pmid + ":aug"
        
        new_entity = x.text[x.start_chars:x.end_chars]    
        #print(orig_entity, new_entity)
        return x


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


    documents = glob(parameters["document_root_dir"] + "/*.txt")

    dfs = []
    for document_path in tqdm(sorted(documents)):
        _, fname = os.path.split(document_path)
        pmid, _ = os.path.splitext(fname)
        with open(document_path) as fp:
            text = fp.read()
            doc = nlp(text)
            for k, sent in enumerate(doc.sents):
                df=entity_extract(nlp(sent.text), pmid,k)
                dfs.append(df)


    df_train = pd.concat(dfs, ignore_index=True)
    #df_train.head(100)

    Mesh_dict_dir= parameters["processed_dict_dir"]

    dictionary_files = [ \
                parameters["cytokine_dict_file"], \
                parameters["tf_dict_file"], \
                parameters["t-lymphocyte_dict_file"], \
                parameters["protein_dict_file"], \
                ]


    global dist_dict
    dist_dict = {}
    for fname in dictionary_files:
        path = os.path.join(Mesh_dict_dir, fname)
        with open(path) as fp:
            lines = [l.strip() for l in fp.readlines()]
            dist_dict[fname] = lines


    lfs = [lf_cytokine_distsv,lf_tf_distsv, lf_t_lymphocyte_distsv]


    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)


    LFAnalysis(L=L_train, lfs=lfs).lf_summary()


    df_train['lf_cytokine_distsv'] = L_train[:,0]
    df_train['lf_tf_distsv'] = L_train[:,1]
    df_train['lf_t_lymphocyte_distsv'] = L_train[:,2]
    #df_train['lf_protein_distsv'] = L_train[:,5]

#    label_model = LabelModel(cardinality=3, verbose=True)
#    label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
#    df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")

    label_model = MajorityLabelVoter(cardinality=3)
    df_train["label"] = label_model.predict(L=L_train)


    df_train_raw = df_train.copy()
    df_train = df_train[df_train.label != ABSTAIN]

    df_train['start_tokens'] = df_train['start_tokens'].astype(np.int64)
    df_train['end_tokens'] = df_train['end_tokens'].astype(np.int64)
    df_train['start_chars'] = df_train['start_chars'].astype(np.int64)
    df_train['end_chars'] = df_train['end_chars'].astype(np.int64)

    
    corpus_dir = parameters["corpus_dir"]
    utils.makedir(corpus_dir)

    df_train_raw.to_csv(os.path.join(corpus_dir, "df_train_raw.csv"))
    df_train.to_csv(os.path.join(corpus_dir, "df_train.csv"))
    df_train_core=df_train[["entities","text","pmid","label"]]
    df_train_core.to_csv(os.path.join(corpus_dir, "df_train_core.csv"))

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))

def data_augumentation(df_train):

    tf_policy = ApplyOnePolicy(n_per_original=2, keep_original=True)
    tf_applier = PandasTFApplier([tf_replace_word_with_synonym_new], tf_policy)
    df_train_augmented = tf_applier.apply(df_train)


    train_text = df_train_augmented.text.tolist()
    X_train = CountVectorizer(ngram_range=(1, 2)).fit_transform(train_text)

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X=X_train, y=df_train_augmented.label.values)


    df_train_augmented.to_csv("df_train_augmented.txt")


    for index, row in df_train_augmented.iterrows():
        text=row['text']
        st = row['start_chars']
        ed = row['end_chars']
        assert(text[st:ed] == row['entities'])
        

    for index, row in df_train_augmented.iterrows():
        print(row)
        text=row['text']
        st = row['start_chars']
        ed = row['end_chars']
        assert(text[st:ed] == row['entities'])
        s = row["start_tokens"]
        e = row["end_tokens"]
        print("text", text)
        words =  [token.text for token in nlp(text)]
        print("words", words)
        tokens = " ".join(words[s:e])
        print("1", text[st:ed], st, ed)
        print("2", tokens, s, e)
        #assert(tokens == text[st:ed])
        

if __name__ == '__main__':                                                                                                                        
    main()




