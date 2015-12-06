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
        
    return labels_predicted
        
        
#%% Fit second model : SVC
def svc(data_train,labels_train,data_test,labels_test,C,show_infos):
    from sklearn.svm import LinearSVC
        
    c=C
    t1 = time()
    clf2 = LinearSVC(C=c)
    y_score = clf2.fit(data_train, labels_train)
    labels_predicted = clf2.predict(data_test)
    t2=time() -t1
    
    if(show_infos == True):
        print "-------------------Vectorizing and fitting the linear SVC took %s"%t2,"sec---------------"
        print "classification report"
        print classification_report(labels_test, labels_predicted)
        print "the accuracy score is :", accuracy_score(labels_test, labels_predicted)
        
    return labels_predicted
    
#%% Fit third model : Logistic Regression
def logRegression(data_train,labels_train,data_test,labels_test,show_infos):
    from sklearn.linear_model import LogisticRegression
    
    
    t1 = time()
    clf3 = LogisticRegression()
    y_score3 = clf3.fit(data_train, labels_train)
    labels_predicted= clf3.predict(data_test)
    t2=time() -t1
    
    if(show_infos == True):
        print "-------------------Vectorizing and fitting the Log-reg took %s"%t2,"sec---------------"
        print "classification report"
        print classification_report(labels_test, labels_predicted)
        print "the accuracy score is :", accuracy_score(labels_test, labels_predicted)
        
    return labels_predicted
    
#%% NBSVM
def nbsvm(data_train,labels_train,data_test,labels_test,C,alpha,beta,show_infos):
    from sklearn.svm import LinearSVC
    import cleantext as ct
    
    #HUGO : la partie "interpolation" grâce au paramètre beta n'est pas encore implémentée
    #Du coup ici c'est comme si on entraine avec beta=1
    c=C
    beta=1
    t1 = time()
    clf4 = LinearSVC(C=c)
    data_train2=ct.nbsvmMatrix(data_train,labels_train,alpha=1)
    y_score4 = clf4.fit(data_train2, labels_train)
    labels_predicted = clf4.predict(data_test)
    t2=time() -t1
    
    if(show_infos == True):
        print "-------------------Vectorizing and fitting the Log-reg took %s"%t2,"sec---------------"
        print "classification report"
        print classification_report(labels_test, labels_predicted)
        print "the accuracy score is :", accuracy_score(labels_test, labels_predicted)
        
    return labels_predicted