
import pandas as pd
import json
import os
import time
from tqdm import tqdm

from utils import utils

CYTOKINE=0
TRANSCRIPTION_FACTOR=1
T_LYMPHOCYTE=2

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

    train_csv = os.path.join(parameters["corpus_dir"], "df_train.csv")

    df_train=pd.read_csv(train_csv, index_col=0)
    df_train=df_train[["entities","start_chars", "end_chars","text","pmid","label"]]
    df_train.reset_index(drop=True, inplace=True)

    df_train=df_train.sort_values(["pmid"])
    df_group=df_train.groupby(["pmid"])

    #print(df_train)

    df_train["entities"]=df_group["entities"].transform(lambda labels: ','.join(str(label) for label in labels))
    df_train["label"]=df_group["label"].transform(lambda labels: ','.join(str(label) for label in labels))
    df_train["start_chars"]=df_group["start_chars"].transform(lambda labels: ','.join(str(label) for label in labels))
    df_train["end_chars"]=df_group["end_chars"].transform(lambda labels: ','.join(str(label) for label in labels))

    #print(df_train.head)


    df_train_clean=df_train.drop_duplicates()
    df_train_clean.reset_index(drop=True, inplace=True)


    df_train_clean.to_csv(os.path.join(parameters["corpus_dir"], "df_train_clean.csv"))

    data_dir = parameters["brat_dataset_dir"]
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
            txt_fp.write("{}\n".format(row["text"]))
        
        with open(ann_file, 'w') as ann_fp:
            for k, (entity, label, char_offset) in  enumerate(zip(entities, labels, char_offsets)):
                
                entity_type = entity_types[int(label)]
                ann_fp.write(f"T{k+1}\t{entity_type} {char_offset[0]} {char_offset[1]}\t{entity}\n")

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))
                
        
if __name__ == '__main__':                                                                                                                        
    main()

