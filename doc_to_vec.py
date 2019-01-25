#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:35:35 2019

@author: adarsh
"""
from gensim.models.doc2vec import Doc2Vec
from os import listdir
from gensim.models.deprecated.doc2vec import LabeledSentence


labels=listdir('files')
contents=[]
for i in labels:
    fp=open('files/'+i)
    contents.append(fp.read())
    
fp = open('files/wiki_world.txt')
contents = fp.read().split('\n')

class LabelDoc(object):
    def __init__(self, para_list):

       self.para_list = para_list
    def __iter__(self):
        for idx, doc in enumerate(self.para_list):
            yield LabeledSentence(words=doc.split(),tags=[idx])
            
itr=LabelDoc(contents)

model = Doc2Vec(size=20, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(itr)
for epoch in range(10):
    model.train(itr,total_examples=model.corpus_count,epochs=100)
    model.alpha -= 0.002 
tokens = "Bismarck".split()

new_vector = model.infer_vector(tokens)
sims = model.docvecs.most_similar([new_vector])

for s in sims:
    print(contents[s[0]])