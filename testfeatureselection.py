import pandas as pd
import numpy as np 

#preprocessing librairies and functions
import cleantext as ct
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#Models librairies
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

#Other
from time import time
import scipy.sparse as sp

import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation

import loadFiles as lf



#%% Load the data and compute alternative features
data, labels = lf.loadLabeled("./train")
train = ct.removehtml(data)

#Create the dictionnary
data=ct.stemTokenize(train)  

#Compute tf-idf including n_grams of size 2 
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), binary=False)
tfidf_matrix = tfidf_vectorizer.fit_transform(data)
data_train, data_test, labels_train, labels_test = train_test_split(tfidf_matrix, labels, test_size = 0.4, random_state  =42)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
score_list = []
features_range = range(5000,tfidf_matrix.shape[1],10000)
for k in range(5000,tfidf_matrix.shape[1],10000) :
	print k 

	ch2 = SelectKBest(chi2, k=k)
	X_new = ch2.fit_transform(data_train, labels_train)
	tfidf_matrix2 = X_new
	tfidf_matrixtest = ch2.transform(data_test)
	clf = MultinomialNB(alpha = .01)
	clf.fit(tfidf_matrix2,labels_train )
	score = clf.score(tfidf_matrixtest, labels_test)
	
	print score
	score_list.append(score)

import matplotlib.pyplot as plt
plt.plot(features_range, score_list)
plt.show()