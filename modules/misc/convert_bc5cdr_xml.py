import os
import sys
import glob
import time
import xmltodict
from tqdm import tqdm

from utils import utils

import pdb

def convert_xml(xml_file, output_dir):

    utils.makedir(output_dir)

    with open(xml_file) as fp:
        data_dict = xmltodict.parse(fp.read())

    #pdb.set_trace()
    documents = data_dict['collection']['document']
    for doc in tqdm(documents):
        docid = doc['id']

        all_annotations = []
        text = ""
        for passage in doc['passage']:
            print(passage['infon'])
            #print(passage['offset'])
            #print(passage['text'], len(passage['text']))
            #print(passage['annotation'])
            try:
                all_annotations.append({"type": passage['infon']["#text"], "ann": passage['annotation']})
                text += passage['text']
            except:
                print("no annotation")
                text += passage['text']
        

        with open(os.path.join(output_dir, f'{docid}.txt'), 'w') as fp:
            fp.write(text)
        

        with open(os.path.join(output_dir, f'{docid}.ann'), 'w') as fp:

            index = 1
            for annotation_dict in all_annotations:
                
                for ann in annotation_dict["ann"]:
                    print("ann", ann)
                    if type(ann) != dict:
                        continue

                    entity = ann['infon'][0]['#text']
                    if type(ann['location']) == dict:
                        offset = int(ann['location']['@offset']) 
                        if annotation_dict["type"] == "abstract":
                            offset -= 1
                        length = int(ann['location']['@length'])
                        ann_txt = ann['text']

                        start_char  = offset
                        end_char = offset + length
                        print(text[start_char:end_char])
                        print(ann_txt)
                        try:
                            assert(text[start_char:end_char] == ann_txt)
                        except:
                            print("ERROR!! assertion error")

                        fp.write(f"T{index}\t{entity} {start_char} {end_char}\t{ann_txt}\n")
                    elif type(ann['location']) == list:

                        offsets = []
                        for loc in ann['location']:
                            offset = int(loc['@offset'])
                            if annotation_dict["type"] == "abstract":
                                offset -= 1
                            length = int(loc['@length'])
                            offsets.append((offset, offset+length))
                        buf = ' '.join([text[offset[0]:offset[1]].strip() for offset in offsets])
                        ann_txt = ann['text']

                        print('buf    ', buf)
                        print('ann_txt', ann_txt)
                        try:
                            assert(buf == ann_txt)
                        except:
                            print("ERROR!! assertion error")

                        offset_buf = ';'.join(["{} {}".format(offset[0],offset[1]) for offset in offsets])
                        fp.write(f"T{index}\t{entity} {offset_buf}\t{ann_txt}\n")


                    index += 1


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


    cdr_corpus_dir = parameters["cdr_corpus_dir"]
    xml_files = sorted(glob.glob(f"{cdr_corpus_dir}/*.xml"))

    for xml_file in xml_files:

        print(xml_file)
        path, fname = os.path.split(xml_file)
        basename, ext = os.path.splitext(fname)
        basename, ext = os.path.splitext(basename)
        
        output_dir = os.path.join(parameters["corpus_dir"], basename)
        convert_xml(xml_file, output_dir)

    print('Done!')
    t_end = time.time()                                                                                                  
    print('Took {0:.2f} seconds'.format(t_end - t_start))


if __name__ == "__main__":
    main()


#cdr_corpus_dir: CDR_Data/CDR.Corpus.v010516
#corpus_dir: data/BC5CDR/corpus
