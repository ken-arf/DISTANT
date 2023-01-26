
import sys
import os
import itertools
import logging
from io import StringIO
import pickle

import numpy as np
import math
import time

from dataclasses import dataclass
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

#from cancer_model import CancerModel
#from cancer_model_all import CancerModel
#from models.cancer_model import CancerModel
from models.evt_model import EvtModel


from tqdm.auto import tqdm

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import get_scheduler
#from huggingface_hub import Repository, get_full_repo_name
from accelerate import Accelerator
import evaluate

from datasets import load_dataset
from datasets import DatasetDict
from datasets.features.features import Sequence
from datasets.features.features import ClassLabel

from utils import utils
from measures import performance

#from loader.eventDataLoader import CustomDataLoader
from loader.eventDataLoader import EventDataLoader

import pdb

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

    eventDataLoader = EventDataLoader(parameters, logger)

    train_dataloader, eval_dataloader, test_dataloader = eventDataLoader.load_data()

    if torch.cuda.is_available() and parameters['gpu'] >= 0:
        device = "cuda"
    else:
        device = "cpu"

    model = EvtModel(eventDataLoader, parameters, logger)
    if parameters['restore_model'] == True:
        model.load_state_dict(torch.load(parameters['restore_model_path'], map_location=torch.device(device)))

    model.config(ner_loss=parameters['ner_loss'], 
                event_loss=parameters['event_loss'], 
                relation_loss=parameters['relation_loss'], 
                attr_loss=parameters['attribute_loss'])


    # evaluate BIO tagging
    tag_mappings = eventDataLoader.get_tag_mappings()
    metrics = [evaluate.load("seqeval") for i in range(len(tag_mappings['bio_tag']))]

    optimizer = AdamW(model.parameters(), lr=float(parameters['train_lr']))

    accelerator = Accelerator()

    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    num_train_epochs = parameters['train_epochs'] 
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        parameters['train_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=parameters['train_num_warmup_steps'],
        num_training_steps=num_training_steps,
    )


    # training model

    progress_bar = tqdm(range(num_training_steps))

    OUTPUT_PATH = parameters['model_dir']
    best_avg_f1 = 0
    patient_count = 0
    steps = 0
    for epoch in range(num_train_epochs):

        # Training
        model.train()
        for batch in train_dataloader:

            #outputs = model(**batch)
            #loss = outputs.loss
            loss, loss_details = model(**batch)
            lossNER, lossEvent, lossRel, lossAttr = loss_details

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description("loss:{:7.2f} (NER:{:7.2f}, EVT:{:7.2f}, REL:{:7.2f}, ATT:{:7.2f}) epoch:{}".
                                     format(loss.item(),
                                            lossNER.item(),
                                            lossEvent.item(),
                                            lossRel.item(),
                                            lossAttr.item(),
                                            epoch))
            steps += 1


        # Evaluation
        running_acc_event = 0
        running_acc_rel = 0
        running_acc_attr = 0
        model.eval()
        for batch_index, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                predictions_list, event_predictions, relation_predictions, attribute_predictions = model.decode(**batch)
                #predictions_list, event_predictions, relation_predictions = model.decode(**batch)

            batch_acc_event = performance.performance_measure_event(event_predictions, logger)
            batch_acc_rel = performance.performance_measure_relation(relation_predictions, logger)
            batch_acc_attr = performance.performance_measure_attribute(attribute_predictions, logger)

            if batch_acc_event and not math.isnan(batch_acc_event):
                running_acc_event += (batch_acc_event - running_acc_event) / (batch_index + 1)
            if batch_acc_rel and not math.isnan(batch_acc_rel):
                running_acc_rel += (batch_acc_rel - running_acc_rel) / (batch_index + 1)
            if batch_acc_attr and not math.isnan(batch_acc_attr):
                running_acc_attr += (batch_acc_attr - running_acc_attr) / (batch_index + 1)

            labels = batch["labels"]

            # split labels into bio_0,...,bio_5

            class_num = len(tag_mappings['bio_tag'])
            labels_split = utils.split_labels(labels, class_num)

            for k, (predictions, labels) in enumerate(zip(predictions_list, labels_split)):

                # Necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                predictions_gathered = accelerator.gather(predictions)
                labels_gathered = accelerator.gather(labels)

                logger.debug("{} predictions".format(k))
                logger.debug(predictions_gathered)
                logger.debug("{} labels".format(k))
                logger.debug(labels_gathered)

                label_names = eventDataLoader.get_bio_labels(f'bio_{k}')

                true_predictions, true_labels = utils.postprocess(predictions_gathered, labels_gathered, label_names)

                metrics[k].add_batch(predictions=true_predictions, references=true_labels)

        average_f1 = 0    
        for k, metric in enumerate(metrics):

            results = metric.compute()
            buffer = StringIO()
            sys.stdout = buffer

            print(
                f"bio_{k}",
                f"epoch {epoch}:", {
                #key: results[f"overall_{key}"]
                key: '{:.03f}'.format(results[f"overall_{key}"])
                for key in ["precision", "recall", "f1", "accuracy"]
                },
            )

            print_out = buffer.getvalue()
            logger.info(print_out)
            sys.stdout = sys.__stdout__

            average_f1 += results["overall_f1"]


        average_f1 /= float(len(metrics))
        msg = f"epoch: {epoch}, steps: {steps}"
        logger.info(msg)
        msg = f"epoch: {epoch}, average_f1: {average_f1:.03f}, event: {running_acc_event:.03f}, rel: {running_acc_rel:.03f}, attr: {running_acc_attr:.03f}"
        #msg = f"epoch: {epoch}, average_f1: {average_f1:.3f}, event: {running_acc_event:.3f}, rel: {running_acc_rel:.3f}, attr: "
        logger.info(msg)


        best_update = False
        if best_avg_f1 < average_f1:
            best_avg_f1 = average_f1
            patient_count = 0
            best_update = True
        else:
            patient_count += 1

        if patient_count > parameters['train_patient_count_limit']:
            msg = "exceed patient_count: {}".format(patient_count)
            logger.info(msg)
            msg = "best_avg_f1: {:.04f}".format(best_avg_f1)
            logger.info(msg)
            break
    

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_PATH, f"model_last.pth"))

        if best_update:
            torch.save(unwrapped_model.state_dict(), os.path.join(OUTPUT_PATH, f"model_best.pth"))

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))

if __name__ == '__main__':                                                                                                                        
    main()


