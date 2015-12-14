# -*- coding: utf-8 -*-
"""
Created on Sun Dec 06 15:45:44 2015

@author: Hugo
"""

from time import time
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#%% Fit first model : Multinomial Naive Bayes Model
def naiveBayes(data_train,labels_train,data_test,labels_test,show_infos):
    
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import cross_validation
    t0 = time()
    clf = MultinomialNB(alpha = .01)
    y_score = clf.fit(data_train, labels_train)
    labels_predicted = clf.predict(data_test)
    t1=time() -t0
    
    if(show_infos == True):
        print "-------------------Vectorizing and fitting the model took %s"%t1,"sec---------------"
        print ""
        print "classification report :"
        print classification_report(labels_test, labels_predicted)
        print "the accuracy score is :", accuracy_score(labels_test, labels_predicted)
        scores = cross_validation.cross_val_score(clf, data_train, labels_train, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
    return labels_predicted
        
        
#%% Fit second model : SVC
def svc(data_train,labels_train,data_test,labels_test,C,show_infos):
    from sklearn.svm import LinearSVC
    from sklearn import cross_validation    
    c=C
    t1 = time()
    clf = LinearSVC(C=c)
    y_score = clf.fit(data_train, labels_train)
    labels_predicted = clf.predict(data_test)
    t2=time() -t1
    
    if(show_infos == True):
        print "-------------------Vectorizing and fitting the linear SVC took %s"%t2,"sec---------------"
        print "classification report"
        print classification_report(labels_test, labels_predicted)
        print "the accuracy score is :", accuracy_score(labels_test, labels_predicted)
        scores = cross_validation.cross_val_score(clf, data_train, labels_train, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
    return labels_predicted
    
#%% Fit third model : Logistic Regression
def logRegression(data_train,labels_train,data_test,labels_test,show_infos):
    from sklearn.linear_model import LogisticRegression
    from sklearn import cross_validation
    
    t1 = time()
    clf = LogisticRegression()
    y_score3 = clf.fit(data_train, labels_train)
    labels_predicted= clf.predict(data_test)
    t2=time() -t1
    
    if(show_infos == True):
        print "-------------------Vectorizing and fitting the Log-reg took %s"%t2,"sec---------------"
        print "classification report"
        print classification_report(labels_test, labels_predicted)
        print "the accuracy score on the test data is :", accuracy_score(labels_test, labels_predicted)
        scores = cross_validation.cross_val_score(clf, data_train, labels_train, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
    return labels_predicted
    
def RNN(data_train, labels_train, data_test, labels_test, n_features):
"""
Adapted from Passage's sentiment.py at
https://github.com/IndicoDataSolutions/Passage/blob/master/examples/sentiment.py
License: MIT
"""
    import numpy as np
    import pandas as pd

    from passage.models import RNN
    from passage.updates import Adadelta
    from passage.layers import Embedding, GatedRecurrent, Dense
    from passage.preprocessing import Tokenizer

    layers = [
        Embedding(size=128, n_features=n_features),
        GatedRecurrent(size=128, activation='tanh', gate_activation='steeper_sigmoid', init='orthogonal', seq_output=False, p_drop=0.75),
        Dense(size=1, activation='sigmoid', init='orthogonal')
    ]
    model = RNN(layers=layers, cost='bce', updater=Adadelta(lr=0.5))
    tokenizer = Tokenizer(min_df=10)
    X = tokenizer.fit_transform(data)
    model.fit(X, labels, n_epochs=10)
    predi = model.predit(data_test).flatten
    labels_predicted = np.ones(len(data_test))
    labels_predicted[predi<0.5] = 0
    
def SGD(data_train,labels_train,data_test,labels_test,show_infos):
    from sklearn.linear_model import SGDClassifier as SGD

    t1 = time()
    clf = SGD(loss='modified_huber')
    y_score3 = clf.fit(data_train, labels_train)
    labels_predicted= clf.predict(data_test)
    t2=time() -t1
    
    if(show_infos == True):
        print "-------------------Vectorizing and fitting the Log-reg took %s"%t2,"sec---------------"
        print "classification report"
        print classification_report(labels_test, labels_predicted)
        print "the accuracy score on the test data is :", accuracy_score(labels_test, labels_predicted)
        scores = cross_validation.cross_val_score(clf, data_train, labels_train, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
    return labels_predicted
    
