#!/usr/bin/env python
# coding: utf-8


import sys
import os
import glob
import spacy
import scispacy
from mosestokenizer import *
import dataclasses


@dataclasses.dataclass
class Sentence:
    text: str
    start_offset: int = 0
    end_offset: int = 0


@dataclasses.dataclass
class Token:
    text: str
    start_offset: int = 0
    end_offset: int = 0


class MyMosesTokenizer:
    def __init__(self):
        self.tokenizer = MosesTokenizer('en')
        self.sentenceSplitter = MosesSentenceSplitter('en')

    def split_sentence(self, doc):
        doc = doc.strip()
        if not type(doc) == list:
            doc = [doc]
        sents = self.sentenceSplitter(doc)

        # print(sents)

        starts = []
        offset = 0
        for sent in sents:
            p = doc[0].index(sent, offset)
            offset += len(sent)
            starts.append(p)

        lengths = [len(sent) for sent in sents]
        ends = [start+leng for start, leng in zip(starts, lengths)]

        result = []
        for txt, start, end in zip(sents, starts, ends):
            s = Sentence(text=txt, start_offset=start, end_offset=end)
            assert (doc[0][start:end] == txt)
            result.append(s)

        return result

    def tokenize(self, text):
        tokens = self.tokenizer(text)

        print(tokens)

        result = []
        offset = 0
        for token in tokens:
            try:
                p = text.index(token, offset)
            except:
                if token == '@-@':
                    token = '-'
                    p = text.index(token, offset)
                else:
                    print("error")

            print(token)
            print(text[p:p+len(token)])

            assert (token == text[p:p+len(token)])
            t = Token(text=token, start_offset=p, end_offset=p+len(token))
            offset = p + len(token)

            result.append(t)

        return result
