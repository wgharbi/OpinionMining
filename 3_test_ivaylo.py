# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:14:17 2015

@author: ivaylo
"""

import re
import numpy as np
from bs4 import BeautifulSoup 
import pandas as pd
from loadFiles import loadLabeled
import matplotlib.pyplot as pl
import scipy.sparse as sp
from sklearn.preprocessing import binarize
from time import time


import cleantext as ct

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
import sklearn.metrics as met
from sklearn.ensemble import BaggingClassifier

# Models
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import SGDClassifier as SGD

#%%

def review_to_wordlist(review):
    '''
    Meant for converting each of the IMDB reviews into a list of words.
    '''
    # First remove the HTML.
    review_text = BeautifulSoup(review,"lxml").get_text()

    # Use regular expressions to only include words.
    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    # Convert words to lower case and split them into separate words.
    words = review_text.lower().split()

    # Return a list of words
    return(words)


#%%

t = time()
# Loading Data
data, Class = loadLabeled('./train')

print"Data loaded: ", int(10*(time()-t))/10.0, "s"


#%%
t = time()
test_ratio = 0.4

#new_features = new_feat_matrix(data)
#for i in range(new_features.shape[1]):
#    col = new_features[:,i]
#    new_features[:,i] = (col-np.min(col))/(np.max(col)-np.min(col))
#new_features = sp.csr_matrix(new_features)

data_all = []
for i in xrange(len(data)):
    data_all.append(" ".join(review_to_wordlist(data[i])))

#tfv = TFIV(min_df=3,  max_features=None, token_pattern
#        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
#        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
#        stop_words = 'english')

tfv = TFIV(ngram_range=(1,3),stop_words=None,binary=False)

tfv.fit(data_all)
old_tf = tfv.transform(data_all)


#count_vect = CountVectorizer(ngram_range=(1,2),binary=False)       
#count_matrix = count_vect.fit_transform(data_all)
#nbsvm_matrix = ct.nbsvmMatrix(count_matrix,Class,alpha=1)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#%%

acc_l = []
#acc_l2 = []

features_taken = range(5000,1000000,50000)

for i in range(len(features_taken)):
    
    
    #final_mat = sp.hstack([nbsvm_matrix,old_tf])
    
    data_train, data_test, labels_train, labels_test = train_test_split(old_tf, Class, test_size = 0.4, random_state  =42)
    
    print"Data preprocessed: ", int(10*(time()-t))/10.0, "s"

    chi = SelectKBest(chi2,k=features_taken[i])
    data_train = chi.fit_transform(data_train,labels_train)
    # pl.data_train = chi.transform(data_train)
    data_test = chi.transform(data_test)
    
    #t = time()
    #base_model = SGD(loss = 'modified_huber')
    #
    ## n_estimator = 100 pour perf max
    #bg_clf = BaggingClassifier(base_estimator=base_model, n_estimators=10)
    #
    #bg_clf.fit(data_train,labels_train)
    #prediction = bg_clf.predict(data_test)
    #print"End of fitting: ", int(10*(time()-t))/10.0, "s"
    #print "Bagging SGD NBSVM err: ", met.accuracy_score(labels_test,prediction)
    #
    
    #t = time()
    #base_model = SGD(loss = 'modified_huber')
    #base_model.fit(data_train,labels_train)
    #prediction = base_model.predict(data_test)
    #print"End of fitting: ", int(10*(time()-t))/10.0, "s"
    #print "Bagging SGD err: ", met.accuracy_score(labels_test,prediction)
    #
    ##%%
    
    #    clf = LR(C=30,penalty = 'l2', dual = True, random_state = 0)    
    #    clf.fit(data_train,labels_train)
    #    print met.accuracy_score(labels_test,clf.predict(data_test))
    #    acc_l2.append(met.accuracy_score(labels_test,clf.predict(data_test)))

    # Multinomial Bayes
    MnBayes = MNB(alpha=0.01, class_prior=None, fit_prior=True)
    MnBayes.fit(data_train, labels_train)
    MNB(alpha=1.0, class_prior=None, fit_prior=True)
    print met.accuracy_score(labels_test,MnBayes.predict(data_test))
    acc_l.append(met.accuracy_score(labels_test,MnBayes.predict(data_test)))
    #print "20 Fold CV Score for Multinomial Naive Bayes: ", np.mean(cross_val_score(MnBayes, data_test, labels_test, cv=20, scoring='accuracy'))
    # This will give us a 20-fold cross validation score that looks at ROC_AUC so we can compare with Logistic Regression. 
    





