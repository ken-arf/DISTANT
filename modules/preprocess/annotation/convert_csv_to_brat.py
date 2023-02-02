
import pandas as pd
import json
import os
import time
import shutil
from tqdm import tqdm

from utils import utils

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer, LancasterStemmer

CYTOKINE=0
TRANSCRIPTION_FACTOR=1
T_LYMPHOCYTE=2

def apply_group(df):

    df['start_chars'] = df['start_chars'].apply(str)
    df['end_chars'] = df['end_chars'].apply(str)
    df['label'] = df['label'].apply(str)
    df_entities = df.groupby(['text', 'pmid'], as_index=False).agg({'entities': ','.join})
    df_start_chars = df.groupby(['text', 'pmid'], as_index=False).agg({'start_chars': ','.join})
    df_end_chars = df.groupby(['text', 'pmid'], as_index=False).agg({'end_chars': ','.join})
    df_label = df.groupby(['text', 'pmid'], as_index=False).agg({'label': ','.join})
    
    df_entities["start_chars"] = df_start_chars["start_chars"]
    df_entities["end_chars"] = df_end_chars["end_chars"]
    df_entities["label"] = df_label["label"]
    
    return df_entities

def get_synonyms(word):
    """Get the synonyms of word from Wordnet."""
    return wn.synsets(word.lower())


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

    train_csv = os.path.join(parameters["corpus_dir"], parameters["train_csv_name"])

    df_train=pd.read_csv(train_csv, index_col=0)
    df_train=df_train[["entities","start_chars", "end_chars","text","pmid","label"]]
    df_train.reset_index(drop=True, inplace=True)

    df_train=df_train.sort_values(["pmid"])

    df_train_clean = apply_group(df_train)
    df_train_clean.reset_index(drop=True, inplace=True)


    basename, ext = os.path.splitext(parameters["train_csv_name"])
    df_train_clean.to_csv(os.path.join(parameters["corpus_dir"], f"{basename}_group_clean.csv"))

    data_dir = parameters["brat_dataset_dir"]
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    utils.makedir(data_dir)

    entity_types = {}
    entity_types[CYTOKINE] = "Cytokine" 
    entity_types[TRANSCRIPTION_FACTOR] = "Transcription_Factor"
    entity_types[T_LYMPHOCYTE] =  "T_Cell"

    rows = df_train_clean.shape[0]

    for index, row in df_train_clean.iterrows():
     #   print(index)
     #   print(row)
        print(f"{index}/{rows}")
        pmid = row["pmid"]
        labels = row["label"].split(",")
        entities = row["entities"].split(",")
        n = len(labels)
        start_chars = row["start_chars"].split(",")
        end_chars = row["end_chars"].split(",")
        char_offsets = [(int(start), int(end)) for start, end in zip(start_chars, end_chars)]
        
        txt_file = os.path.join(data_dir, f"PMID_{pmid}.txt")
        ann_file = os.path.join(data_dir, f"PMID_{pmid}.ann")
        
        with open(txt_file, 'w') as txt_fp:
            txt_fp.write("{}\n".format(row["text"].strip()))
            doc_len = len(row["text"].strip())
        
        
        with open(ann_file, 'w') as ann_fp:
            for k, (entity, label, char_offset) in  enumerate(zip(entities, labels, char_offsets)):

                if len(get_synonyms(entity)) != 0:
                    continue
                    
                start_char = char_offset[0]
                end_char = char_offset[1]
                if end_char > doc_len:
                    end_char = doc_len

                entity_type = entity_types[int(label)]
                ann_fp.write(f"T{k+1}\t{entity_type} {start_char} {end_char}\t{entity}\n")

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))
                
        
if __name__ == '__main__':                                                                                                                        
    main()

