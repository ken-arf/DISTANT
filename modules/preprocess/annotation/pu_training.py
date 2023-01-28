import pandas as pd
import spacy
import scispacy
import os
import time
import pickle
import numpy as np
import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import get_scheduler
from accelerate import Accelerator
import evaluate


from tqdm.auto import tqdm

from utils import utils

from pu_dataloader import PU_Dataloader 
from pu_model import PU_Model 

from measure import performance
import pdb

pd.set_option('display.max_colwidth', None)


def train(df_train_pos, df_train_neg, parameters):

    # logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.ERROR)

    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"))

    #handler2 = logging.FileHandler(filename="test.log")
    #handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(handler1)
    #logger.addHandler(handler2)

    pu_dataloader = PU_Dataloader(df_train_pos, df_train_neg, parameters)

    train_dataloader, valid_dataloader, test_dataloader = pu_dataloader.load_data()

    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    pu_model = PU_Model(pu_dataloader, parameters, logger)

    if parameters['restore_model'] == True:
        pu_model.load_state_dict(torch.load(parameters['restore_model_path'], map_location=torch.device(device)))

    optimizer = AdamW(pu_model.parameters(), lr=float(parameters['train_lr']))
    accelerator = Accelerator()
    pu_model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        pu_model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )


    num_train_epochs = parameters['train_epochs']
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    print("num_update_steps_per_epoch", num_update_steps_per_epoch)
    print("num_train_epochs", num_train_epochs)
    print("num_training_steps", num_training_steps)

    lr_scheduler = get_scheduler(
        parameters['train_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=parameters['train_num_warmup_steps'],
        num_training_steps=num_training_steps,
    )

    # training model

    progress_bar = tqdm(range(num_training_steps))

    OUTPUT_PATH = parameters['model_dir']
    best_acc = 0
    patient_count = 0
    steps = 0
    for epoch in range(num_train_epochs):
        break

        # Training
        pu_model.train()
        for batch in train_dataloader:

            loss = pu_model(**batch)
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description("loss:{:7.2f} epoch:{}".format(loss.item(),epoch))
            steps += 1

        # Evaluation

        progress_bar_valid = tqdm(range(len(valid_dataloader)))
        running_acc = 0
        pu_model.eval()
        for batch_index, batch in tqdm(enumerate(valid_dataloader)):
            with torch.no_grad():
                predictions, probs = pu_model.decode(**batch)
            labels = batch["labels"].detach().cpu().numpy()
            batch_acc = performance.performance_acc(predictions, labels, logger)
            running_acc += (batch_acc - running_acc) / (batch_index + 1)
            progress_bar_valid.update(1)
            progress_bar_valid.set_description("running_acc:{:.2f} epoch:{}".format(running_acc, epoch))
        print("running_acc: {:.2f}".format(running_acc))

        best_update = False
        if best_acc < running_acc:
            best_acc = running_acc
            # save model
            patient_count = 0
            best_update = True
        else:
            patient_count += 1

        if patient_count > parameters["max_patient_count"]:
            print("Exceed max patient count. Force to exit")
            break

        # prepare saving model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(pu_model)

        OUTPUT_PATH = parameters['model_dir']

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_PATH, f"model_last.pth"))

        if best_update:
            torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_PATH, f"model_best.pth"))

    print("best_acc: {:.2f}".format(best_acc))


    # load best model
    pu_model.load_state_dict(torch.load(parameters['restore_model_path'], map_location=torch.device(device)))

    # testing
    progress_bar_test = tqdm(range(len(test_dataloader)))
    probs_list = []
    pu_model.eval()
    for batch_index, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            predictions, probs = pu_model.decode(**batch)
        labels = batch["labels"].detach().cpu().numpy()
        batch_acc = performance.performance_acc(predictions, labels, logger)
        progress_bar_test.update(1)
        progress_bar_test.set_description("acc:{:.2f}".format(batch_acc))
        probs_list.append(probs)

    neg_probs = np.vstack(probs_list)

    return neg_probs


def setup_dataset(step, parameters):

    frac_spy = 0.05
    corpus_dir = parameters["corpus_dir"]

    seed = parameters["seed"]

    print("load dataset", os.path.join(corpus_dir,"df_train_raw.csv"))
    df_train_raw = pd.read_csv(os.path.join(corpus_dir,"df_train_raw.csv"), index_col=0)

    df_train_raw = df_train_raw.dropna()

    df_train_raw.reset_index(drop=True, inplace=True)


    df_unknown = df_train_raw[df_train_raw["label"]==-1]
    df_train_0 =  df_train_raw[df_train_raw["label"]==0]
    df_train_1 =  df_train_raw[df_train_raw["label"]==1]
    df_train_2 =  df_train_raw[df_train_raw["label"]==2]


    df_train_0_spy = df_train_0.sample(frac=frac_spy, random_state=seed)
    df_train_1_spy = df_train_1.sample(frac=frac_spy, random_state=seed)
    df_train_2_spy = df_train_2.sample(frac=frac_spy, random_state=seed)



    indexes = [k for k in df_train_0.index.tolist() if k not in df_train_0_spy.index.tolist()]
    df_train_0_diff = df_train_0.loc[indexes]

    indexes = [k for k in df_train_1.index.tolist() if k not in df_train_1_spy.index.tolist()]
    df_train_1_diff = df_train_1.loc[indexes]

    indexes = [k for k in df_train_2.index.tolist() if k not in df_train_2_spy.index.tolist()]
    df_train_2_diff = df_train_2.loc[indexes]

    df_train_pos_step = pd.concat([df_train_0_diff, df_train_1_diff, df_train_2_diff])
    df_train_neg_step = pd.concat([df_unknown, df_train_0_spy, df_train_1_spy, df_train_2_spy])


    print("df_train_raw", df_train_raw.shape)
    print("df_train_pos_step", df_train_pos_step.shape)
    print("df_train_neg_step", df_train_neg_step.shape)

    df_train_neg_step.rename(columns={"label": "olabel"}, inplace=True)
    labels = [-1] * df_train_neg_step.shape[0]
    df_train_neg_step["label"] = labels

    df_train_pos_step["olabel"] = df_train_pos_step.loc[:,"label"]

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

    df_train_pos, df_train_neg = setup_dataset(step, parameters)
    neg_probs = train(df_train_pos, df_train_neg, parameters)

    model_dir = parameters["model_dir"]
    df_train_pos.to_csv(os.path.join(model_dir, "train_pos.csv"))
    df_train_neg.to_csv(os.path.join(model_dir, "train_neg.csv"))

    with open(os.path.join(model_dir, "neg_prob.pkl"), 'bw') as fp:
        pickle.dump(neg_probs, fp)

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':                                                                                                                        
    main()



