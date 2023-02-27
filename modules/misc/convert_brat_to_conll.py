#!/usr/bin/env python
# coding: utf-8


import sys
import os
import re
import glob
import spacy
import scispacy
import pandas as pd
import time
from utils import utils



#nlp = spacy.load("en_core_sci_lg")
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("sentencizer")



def annotate(tokens, offsets, ann_data_list):
    print(tokens)
    print(offsets)
    print(ann_data_list)
    starts = [off[0] for off in offsets]
    ends =  [off[1] for off in offsets]
    bio_labels = ['O'] * len(tokens)
    
    for ann_data in ann_data_list:
        text = ann_data['text']
        offsets = ann_data['offsets']
        ent = ann_data['entity']
        # continuous label
        if len(offsets) == 1:
            start_char = offsets[0][0]
            end_char = offsets[0][1]
            # assign BIO labels
            if start_char in starts and end_char in ends:
                start_idx = starts.index(start_char)
                end_idx = ends.index(end_char)
                #print(start_idx, end_idx)
                for k in range(start_idx, end_idx+1):
                    if k == start_idx:
                        bio_labels[k] = f'B_{ent}'
                    else:
                        bio_labels[k] = f'I_{ent}'
                #print(bio_labels)
        # uncontinuous label
        else:
            for I, offset in enumerate(offsets):
                start_char = offset[0]
                end_char = offset[1]
                # assign BIO labels
                if start_char in starts and end_char in ends:
                    start_idx = starts.index(start_char)
                    end_idx = ends.index(end_char)
                    for k in range(start_idx,end_idx+1):
                        if k == start_idx and I == 0:
                            bio_labels[k] = f'B_{ent}'
                        elif k == start_idx:
                            bio_labels[k] = f'S_{ent}'
                        else:
                            bio_labels[k] = f'I_{ent}'
                            
    #print(bio_labels)
    return bio_labels
                

    

def convert(file, ann, output_path):
    with open(file) as fp:
        text = fp.read()
    with open(ann) as fp:
        lines = [line.strip('\n') for line in fp.readlines()]
        
    ann_data_list = []
    for line in lines:
        print(line)
        ann_data = {}
        ann_txt = line.split('\t')[-1]
        entity = line.split('\t')[1].split(' ')[0]
        matches = re.findall(r"(\d+) (\d+)", line)
        offsets = []
        for m in matches:
            start_char = int(m[0])
            end_char = int(m[1])
            offsets.append((start_char, end_char))
        ann_data['text'] = ann_txt
        ann_data['entity'] = entity
        ann_data['offsets'] = offsets
        ann_data_list.append(ann_data)
        

    with open(output_path, 'w') as fp:

        doc = nlp(text)
        for snt in doc.sents:
            #print(snt)
            tokens = [token for token in snt]
            offsets = [(token.idx, token.idx + len(token.text)) for token in snt]
            labels = annotate(tokens, offsets, ann_data_list)
            for token, label in zip(tokens, labels):
                fp.write(f"{token}\t{label}\n")
            fp.write("\n")
            

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

    
    root = parameters["corpus_dir"]
                    
    files = sorted(glob.glob(f"{root}/*.txt"))
    ann_files = [file.replace(".txt", ".ann") for file in files]
    for file, ann in zip(files, ann_files):
        print(file)
        path, fname = os.path.split(file)
        basename, ext = os.path.splitext(fname)

        output_path = os.path.join(path, f"{basename}.coll")
        convert(file, ann, output_path)



    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))
                
        
if __name__ == "__main__":
    main()
    

