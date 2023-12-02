import os
import sys

import requests
import pandas as pd
import json
from lxml import etree
import time
from tqdm import tqdm

from utils import utils

import pdb

from Bio.Entrez import efetch
from Bio import Entrez
Entrez.email = 'yano0828@gmail.com'


def get_abstract(pmid):

    try:
        handle = efetch(db='pubmed', id=pmid,
                        retmode='text', rettype='abstract')
        buff = handle.read()
    except:
        print("exception efetch")
        return ""

    abst_lines = []

    mode = "none"
    for k, txt in enumerate(buff.split('\n')):
        if txt.startswith("Author information"):
            mode = "author"
        elif len(txt) == 0 and mode == "author":
            mode = "abst"
        elif mode == "abst" and len(txt) == 0:
            mode = "none"
        elif mode == "abst" and len(txt) > 0:
            abst_lines.append(txt)

    abst_text = "".join(abst_lines)

    return abst_text


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

    pmid_path = os.path.join(parameters['pubmed_pmid_dir'], 'pmid.txt')

    with open(pmid_path) as fp:
        pmids = fp.read()

    pmids = [pmid for pmid in pmids.split('\n')]

    utils.makedir(parameters['pubmed_extract_dir'])
    for pmid in tqdm(pmids):
        abst_file = os.path.join(
            parameters['pubmed_extract_dir'], "{}.txt".format(pmid))

        # if os.path.exists(abst_file):
        #    print(abst_file, " exist, skip")
        #    continue
        abst = get_abstract(pmid)
        if len(abst) <= 250:
            continue

        with open(abst_file, 'w') as fp:
            fp.write("{}\n".format(abst))
        time.sleep(0.2)

    print('Done!')
    t_end = time.time()
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == '__main__':
    main()
