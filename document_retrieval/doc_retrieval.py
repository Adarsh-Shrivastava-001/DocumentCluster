#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:51:57 2018

@author: rushikesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading the dataset
dataset = pd.read_csv('people.csv')

#Variable person
person = dataset[dataset['Name']== 'Kareena Kapoor Khan']
person['Info']

#Just a testcase
ajay = dataset[dataset['Name']== 'Saif Ali Khan']
ajay['Info']

X = dataset.iloc[:,1:].values




#Tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer().fit(X['Info'])
len(vect.get_feature_names())


#Vectorizing all entities
ajay_vectorized = vect.transform(ajay['Info'])
person_vectorized = vect.transform(person['Info'])

X_vectorized = vect.transform(X['Info'])


#Manually calculating the distance

from sklearn.metrics.pairwise import cosine_similarity
distance = cosine_similarity(person_vectorized[0], ajay_vectorized[0])




#Building a knn model

from sklearn.neighbors import NearestNeighbors

#Since the tf-idf already normalizes the vector we can use euclidean distance also
classifier = NearestNeighbors(n_neighbors=5)
classifier.fit(X_vectorized)
ans = classifier.kneighbors(person_vectorized[0], n_neighbors=5)
ind = ans[1]


#Printing the closest five
for i in ind:
    print(X[i][:,0])





