


# coding: utf-8
import gensim
import torch
import torch.nn as nn

import pdb

pdb.set_trace()

# Load word2vec pre-train model
model = gensim.models.KeyedVectors.load_word2vec_format('PubMed-w2v.bin', binary=True)
#model = gensim.models.Word2Vec.load('./word2vec_pretrain_v300.model')
weights = torch.FloatTensor(model.vectors)


# Build nn.Embedding() layer
embedding = nn.Embedding.from_pretrained(weights)
embedding.requires_grad = False


# Query
query = 'pain'
query_id = torch.tensor(model.get_index('pain'))

gensim_vector = torch.tensor(model[query])
embedding_vector = embedding(query_id)

print(gensim_vector==embedding_vector)
