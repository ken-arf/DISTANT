#!/usr/bin/env python
# coding: utf-8


import sys
import os
import re
import glob
import pandas as pd




def check(file, ann):
    with open(file) as fp:
        text = fp.read()
    with open(ann) as fp:
        lines = [line.strip('\n') for line in fp.readlines()]
    
    for line in lines:
        #print(line)
        ann_txt = line.split('\t')[-1]
        matches = re.findall(r"(\d+) (\d+)", line)
        buf = []
        for m in matches:
            start_char = int(m[0])
            end_char = int(m[1])
            txt = text[start_char:end_char]
            buf.append(txt)
        txt = ' '.join(buf)
        if ann_txt != txt:
            print("line:", line)
            print("ann_txt:", ann_txt)
            print("txt", txt)
    

def main():

    root="../data/BC5CDR/processed_corpus"
    dirs = glob.glob(f'{root}/*')

    for dir in dirs:
        print(dir)
        files = sorted(glob.glob(f"{dir}/*.txt"))
        ann_files = [file.replace(".txt", ".ann") for file in files]
        for file, ann in zip(files, ann_files):
            print(file)
            check(file, ann)



if __name__ == "__main__":
    main()

