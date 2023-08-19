from nltk.stem import PorterStemmer, WordNetLemmatizer, LancasterStemmer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
import os
import sys

import nltk
nltk.download("wordnet", quiet=True)
nltk.download('punkt')

# from snorkel.augmentation import ApplyOnePolicy, PandasTFApplier
# from snorkel.augmentation import transformation_function
# from snorkel.labeling import PandasLFApplier, LFApplier, LFAnalysis, labeling_function
# from snorkel.analysis import get_label_buckets
# from snorkel.labeling.model import LabelModel
# from snorkel.labeling.model import MajorityLabelVoter
# from snorkel.labeling.model import LabelModel
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression


class EntityItem:
    def __init__(self, name, label, index, dict_name, parameters):

        self.name = name
        self.label = label
        self.index = index
        self.dict_name = dict_name
        self.params = parameters

        self.lemmatizer = WordNetLemmatizer()
        self.max_dist = 0

        self.prepare_synonym_table()
        self.load_dict()

    def load_dict(self):

        self.dictionary_items = []

        dict_dir = self.params["processed_dict_dir"]
        path = os.path.join(dict_dir, self.dict_name)
        with open(path) as fp:
            lines = [l.strip() for l in fp.readlines()]
            lines_lower = [l.strip().lower() for l in fp.readlines()]
            self.dictionary_items = lines + lines_lower

    def prepare_synonym_table(self):

        self.synonym_table = {
            'lymphocyte': 'cell',
            'lymphocyte': 'lymph cell',
            'cell': 'lymphocyte',
            '+': 'positive',
            '-': 'negative',
        }

        greek_name = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron',
                      'pi', 'pho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']
        greek_letter = list('αβγδεζηθικλμνξοπρστυφχψω')

        greek_translate = {l: w for l, w in zip(greek_letter, greek_name)}

        self.synonym_table.update(greek_translate)

    def min_edit_distance(self, ref, src):

        ref = ' '.join([self.lemmatizer.lemmatize(w.lower())
                       for w in ref.split()])
        src = ' '.join([self.lemmatizer.lemmatize(w.lower())
                       for w in src.split()])

        min_l = ed.eval(ref, src)

        # synonym word exchange
        for k, v in self.synonym_table.items():
            if not k in src:
                continue

            src2 = src.replace(k, v)
            l = ed.eval(ref, src2)
            if l < min_l:
                min_l = l
                src = src2

        if min_l > 0:
            ref = re.sub(r'\W', '', ref)
            src = re.sub(r'\W', '', src)
            if src == ref:
                min_l = 0
