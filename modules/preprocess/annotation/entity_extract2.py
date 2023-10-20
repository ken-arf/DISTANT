import sys
import os
from glob import glob

import spacy
from scispacy.abbreviation import AbbreviationDetector
import scispacy
from spacy.lang.en import English


from preprocess.segmentation.predict2 import EntityExtraction
from preprocess.segmentation.predict2 import Entity
from preprocess.segmentation.make_dataset import tokenize
from preprocess.segmentation.make_dataset import sentence_split

import pdb


class ExtractEntityCandidate:

    def __init__(self, config_file):

        self.entityExtraction = EntityExtraction(config_file)
        # self.nlp = spacy.load("en_core_sci_lg")
        self.nlp = spacy.load("en_core_sci_sm")
        self.nlp.add_pipe("sentencizer")

    def check_overlap(self, entities, term):

        for ent in entities:
            if term.start_char <= ent.end_char and term.end_char >= ent.start_char:
                return True
        return False

    def extract_candiate(self, text, custom_model=True, scipy_model=True):

        entities = []

        # entities extracted from trained custom model
        if custom_model:
            entities += self.entityExtraction.get_entities(text)

        # entities extracted from scispacy
        if scipy_model:
            for term in self.nlp(text).ents:
                if not self.check_overlap(entities, term):
                    ent = Entity(text=term.text, start=term.start, end=term.end,
                                 start_char=term.start_char, end_char=term.end_char)
                    entities.append(ent)

        entities = sorted(entities, key=lambda x: x.start_char)
        return entities


if __name__ == "__main__":

    config_dir = "/Users/kenyano/WORK/AIST/Immunology/configs"
    config_file = "immunology_segmentation_predict.yaml"

    entityExtraction = ExtractEntityCandidate(
        os.path.join(config_dir, config_file))

    text_dir = "/Users/kenyano/WORK/AIST/Immunology/data/Mesh/PubMed/Immunology/extract"
    files = sorted(glob(f"{text_dir}/*.txt"))

    # files = ["/Users/kenyano/WORK/AIST/Immunology/data/Mesh/PubMed/Immunology/extract/10450520.txt"]
    files = [
        "/Users/kenyano/WORK/AIST/Immunology/data/Mesh/PubMed/Immunology/extract/34247018.txt"]

    for file in files:
        with open(file) as fp:
            text = fp.read().strip('\n')

        print("###", file, "###")
        sents = sentence_split(text)

        for sent in sents:
            print("-"*10)
            print(sent)
            entities = entityExtraction.extract_candiate(sent)
            for k, ent in enumerate(entities):
                print(k, ent)
