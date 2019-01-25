#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 21:16:45 2019

@author: adarsh
"""
#
#import gensim.downloader as api
#
#info = api.info()  # show info about available models/datasets
#model = api.load("glove-wiki-gigaword-200")  # download the model and return as object ready for use
#model.most_similar("good")
#
#


#from gensim.models import KeyedVectors, Word2Vec









import gensim
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)  


import numpy as np
import re

dim=300

def conv2vec(para):
    words=para.split(' ')
    num=0
    res=np.zeros((dim,1))
    for word in words:
        try:
            word_vec=model[word.lower()]
            res=res+word_vec
            num=num+1
        except:
            pass
    return res/num

def calc_dist(para_vec, arr, dis='cosine'):
    if dis=='euc':
        dist_arr = np.sum(np.square(arr-para_vec.T), axis=1)
    elif dis=='cos':
        dist_arr = np.dot(arr,para_vec)
        
    ind_arr = np.sort(dist_arr)
    return ind_arr


def clean(doc):
    doc=re.sub('[^a-zA-Z- ]',' ', doc)
    return doc
    


doc=open('ww2').read()
doc=clean(doc).split(' ')
doc_vocab={}
for i in doc:
    if doc_vocab.get(i)!=None:
        doc_vocab[i]=doc_vocab[i]+1
    else:
        doc_vocab[i]=1
        
