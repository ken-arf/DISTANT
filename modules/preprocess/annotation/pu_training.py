import pandas as pd
import spacy
import scispacy
import os
import time

from utils import utils

from train import PUTrain 


nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("sentencizer")
seed=68

pd.set_option('display.max_colwidth', None)


def train(df_train_pos, df_train_neg, parameters):

    trainer = PUTrain(df_train_pos, df_train_neg, parameters)


def setup_dataset(step, parameters):


    corpus_dir = parameters["corpus_dir"]

    df_train_raw = pd.read_csv(os.path.join(corpus_dir,"df_train_raw.csv"), index_col=0)

    df_train_raw = df_train_raw.dropna()

    df_train_raw.reset_index(drop=True, inplace=True)


    df_unknown = df_train_raw[df_train_raw["label"]==-1]
    df_train_0 =  df_train_raw[df_train_raw["label"]==0]
    df_train_1 =  df_train_raw[df_train_raw["label"]==1]
    df_train_2 =  df_train_raw[df_train_raw["label"]==2]


    df_train_0_sample = df_train_0.sample(frac=0.05, random_state=seed)
    df_train_1_sample = df_train_1.sample(frac=0.05, random_state=seed)
    df_train_2_sample = df_train_2.sample(frac=0.05, random_state=seed)



    indexes = [k for k in df_train_0.index.tolist() if k not in df_train_0_sample.index.tolist()]
    df_train_0_diff = df_train_0.loc[indexes]

    indexes = [k for k in df_train_1.index.tolist() if k not in df_train_1_sample.index.tolist()]
    df_train_1_diff = df_train_1.loc[indexes]

    indexes = [k for k in df_train_2.index.tolist() if k not in df_train_2_sample.index.tolist()]
    df_train_2_diff = df_train_2.loc[indexes]

    df_train_pos_step = pd.concat([df_train_0_diff, df_train_1_diff, df_train_2_diff])
    df_train_neg_step = pd.concat([df_unknown, df_train_0_sample, df_train_1_sample, df_train_2_sample])


    df_train_neg_step.rename(columns={"label": "olabel"}, inplace=True)
    labels = [-1] * df_train_neg_step.shape[0]
    df_train_neg_step["label"] = labels

    return df_train_pos_step, df_train_neg_step

def main():

    # set config path by command line
    inp_args = utils._parsing()                                                                                            
    config_path = getattr(inp_args, 'yaml')
    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    # print config
    utils._print_config(parameters, config_path)

    step = 1

    # check running time
    t_start = time.time()                                                                                                  

    while True:
        df_train_pos, df_train_neg = setup_dataset(step, parameters)
        train(df_train_pos, df_train_neg, parameters)
        break

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))



if __name__ == '__main__':                                                                                                                        
    main()



