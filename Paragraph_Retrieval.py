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


from pyemd import emd
from gensim.similarities import WmdSimilarity
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)  


import numpy as np
import re

dim=300

def conv2vec(para, wts):
    words=para.split(' ')
    res=np.zeros((dim,))
    for word in words:
        try:
            word_vec=model[word.lower()]
            res=res+word_vec*wts[word]
        except:
            pass
    return res

def calc_dist(para_vec, arr, dis='cos'):
    dist_arr=None
    if dis=='euc':
        dist_arr = np.sum(np.square(arr-para_vec.T), axis=1)
    elif dis=='cos':
        dist_arr = np.dot(arr,para_vec)
        
    ind_arr = np.argsort(dist_arr)
    return ind_arr


def clean(doc):
    doc=re.sub('[^a-zA-Z-\n ]',' ', doc)
    doc_=''
    for i in range(len(doc)-1):        
        if doc[i]==' ' and doc[i+1]==' ':
            pass
        else:
            doc_=doc_+doc[i]
            
    return doc_
    
def split_in_para(doc):
    return doc.split('\n')

def inverse_softmax(vocab, sent):
    sent=sent.split(' ')
    sent = set(sent)
    K=0
    denominator =0
    weights = {}
    for word in sent:
        try: 
            denominator+= np.exp(-vocab[word]*K)
        except:
            pass

    for word in sent:
        if word not in weights:
            try:
                weights[word]= np.exp(-vocab[word]*K)/denominator
            except:
                pass
        # else:   #theoretical decision to take frequency of word in entire document irrespective of the sentence.
        #     weights[word]= weights[word]*(np.exp(-1))

    return weights  


    
    

doc=open('files/ww2').read().lower()
doc=clean(doc)
words=doc.split(' ')
doc_vocab={}
doc_vocab[' ']=0
doc_vocab['-']=0
for i in words:
    if doc_vocab.get(i)!=None:
        doc_vocab[i]=doc_vocab[i]+1
    else:
        doc_vocab[i]=1
        

paras_=split_in_para(doc)
paras=[]
for i in paras_:
    if i!='':
        paras.append(i)
paras_vec=[conv2vec(para,inverse_softmax(doc_vocab, para)) for para in paras]

result = []
for para in paras:
    similarity = model.wmdistance(query, para)
    result.append(similarity)
    
ind_arr = np.argsort(result)
    
        

        
query="world war ends"
qu_wts=inverse_softmax(doc_vocab,query)
query_vec=conv2vec(query, qu_wts)
        
aa=calc_dist(query_vec, paras_vec)