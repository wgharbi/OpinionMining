# -*- coding: utf-8 -*-


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
        print "-------------------Vectorizing and fitting the MultinomialNB took %s"%t1,"sec---------------"
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
    
def StochasGD(data_train,labels_train,data_test,labels_test,show_infos):
    from sklearn.linear_model import SGDClassifier as SGD
    from sklearn import cross_validation    

    t1 = time()
    clf = SGD(loss='modified_huber')
    y_score3 = clf.fit(data_train, labels_train)
    labels_predicted= clf.predict(data_test)
    t2=time() -t1
    
    if(show_infos == True):
        print "-------------------Vectorizing and fitting SGD took %s"%t2,"sec---------------"
        print "classification report"
        print classification_report(labels_test, labels_predicted)
        print "the accuracy score on the test data is :", accuracy_score(labels_test, labels_predicted)
        scores = cross_validation.cross_val_score(clf, data_train, labels_train, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
    return labels_predicted

def RandomNbSGD(data_train,labels_train,data_test,labels_test,show_infos,n_estima=10):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.linear_model import SGDClassifier as SGD
    from sklearn import cross_validation

    t1 = time()
    base_model = SGD(loss = 'modified_huber')
    # n_estimator = 100 pour perf max
    clf = BaggingClassifier(base_estimator=base_model, n_estimators=n_estima)
    y_score3 = clf.fit(data_train, labels_train)
    labels_predicted= clf.predict(data_test)
    t2=time() -t1
    
    if(show_infos == True):
        print "-------------------Vectorizing and fitting the SGD with a modified_huber loss took %s"%t2,"sec---------------"
        print "classification report"
        print classification_report(labels_test, labels_predicted)
        print "the accuracy score on the test data is :", accuracy_score(labels_test, labels_predicted)
        scores = cross_validation.cross_val_score(clf, data_train, labels_train, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    
#%%
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
from sklearn.preprocessing import binarize
import numpy as np
    
class NBmatrix(BaseEstimator, TransformerMixin):
   
    
    def __init__(self, alpha,bina,n_jobs):
        self.alpha = alpha
        self.bina = bina
        self.n_jobs = 1
        self.r = []

    def fit(self, X, y):
        alpha = self.alpha
        nb_doc, voc_length = X.shape
        pos_idx=[y==1][0].astype(int)
        neg_idx=[y==0][0].astype(int)
        #Store the indicator vectors in sparse format to accelerate the computations
        pos_idx=sp.csr_matrix(pos_idx.T)
        neg_idx=sp.csr_matrix(neg_idx.T)
        #Use sparse format dot product to get a weightning vector stored in sparse format
        alpha_vec=sp.csr_matrix(alpha*np.ones(voc_length))
        p = (alpha_vec + pos_idx.dot(X)) 
        norm_p = p.sum()
        p = p.multiply(1/norm_p)
        #print p.toarray()
        q = (alpha_vec + neg_idx.dot(X))
        norm_q = q.sum()
        q = q.multiply(1/norm_q)
        #print q.toarray()
        
        ratio = sp.csr_matrix(np.log((p.multiply(sp.csr_matrix(np.expand_dims(q.toarray()[0]**(-1),axis=0)))).data))
        #print ratio.toarray()
        self.r = ratio #Stock the ratio vector to re-use it for transforming unlablled data
        return self

    def transform(self, X):
        
        
        #If the binarize option is set to true, we need now to recompute "f", our binarized word counter
        if(self.bina == True):
            f_hat = binarize(X, threshold = 0.0)
        else :
            f_hat=X
        
        f_tilde = f_hat.multiply(self.r)
        return f_tilde
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X,y)

