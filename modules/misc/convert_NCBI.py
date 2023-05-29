import os
import sys
import glob
import time
import xmltodict
from tqdm import tqdm

from utils import utils

import pdb

def convert_text(file, output_dir):

    utils.makedir(output_dir)

    with open(file) as fp:
        data = fp.read()
    samples = data.split('\n\n')

    print(output_dir)
    for sample in samples:
        lines = sample.strip().split('\n')
        #pdb.set_trace()
        annotations = []
        for k, line in enumerate(lines):
            line = line.strip()
            print(k, line)
            if k==0:
                fields = line.split('|')
                fid1 = fields[0]
                kind = fields[1]
                assert (kind=='t')
                cont1 = fields[2].strip()
            elif k==1:
                fields = line.split('|')
                fid2 = fields[0]
                assert (fid1==fid2)
                kind = fields[1]
                assert (kind=='a')
                cont2 = fields[2].strip()
            else:
                fields = line.split('\t')
                fid3 = fields[0]
                assert(fid1 == fid3)
                char_start = int(fields[1])
                char_end = int(fields[2])
                mention = fields[3]
                entity_type = fields[4]
                link_name = fields[5]
                annotations.append((mention, char_start, char_end, entity_type, link_name))
                
        txt_file = os.path.join(output_dir, f"{fid1}.txt")
        with open(txt_file, 'w') as fp:
            fp.write("{}\n{}".format(cont1, cont2))
        ann_file = os.path.join(output_dir, f"{fid1}.ann")
        with open(ann_file, 'w') as fp:
            for ann in annotations:
                fp.write(f'{ann[0]}\t{ann[1]}\t{ann[2]}\tDisease\t{ann[3]}\t{ann[4]}\n')
                


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


    ncbi_dir = parameters["NCBI_corpus_dir"]
    corpus_files = sorted(glob.glob(f"{ncbi_dir}/*.txt"))

    for file in corpus_files:
        print(f"processing {file}")
        path, fname = os.path.split(file)
        basename, ext = os.path.splitext(fname)
        basename, ext = os.path.splitext(basename)
        
        output_dir = os.path.join(parameters["corpus_dir"], basename)
        convert_text(file, output_dir)

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == "__main__":
    main()


#cdr_corpus_dir: CDR_Data/CDR.Corpus.v010516
#corpus_dir: data/BC5CDR/corpus
