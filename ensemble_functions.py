# -*- coding: utf-8 -*-


import re
import numpy as np
from bs4 import BeautifulSoup 
import pandas as pd
from loadFiles import loadLabeled, loadUknown
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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
# Models
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

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

def first_layer(basic_model, data_train, labels_train,data_test,labels_test):
    """
    Compute the probas for data_test for every model in basic model.
    
    input :
        basic_model = list of (clf,"clf_name") containing classifiers
        data_train = data used to fit the models, array like
        labels_train = labels associated to data_train, np.array
        data_test = data used to test the models, array like
        labels_train = labels associated to data_test, np.array

    output :
        probas = Array containing the probas for models in basic models, np.array((len(data_test,len(basic_model))))                      
        best_score = best score from all the models, float
        best_model = model corresponding to best score, str
    """
    nb_basic = len(basic_model)
    probas = np.zeros((data_test.shape[0],nb_basic))
    score_list = []
    for i in xrange(nb_basic):
        model = basic_model[i][0]
        #model_name = basic_model[i][1]
        model.fit(data_train,labels_train)
        score = met.precision_score(labels_test,model.predict(data_test))
        score_list.append(score)
        probas[:,i] = model.predict_proba(data_test)[:,1]
    best_score = max(score_list)
    best_model = basic_model[np.argmax(score_list)][1]      
    return probas, best_score, best_model


def second_layer(advced_model, proba_array_train,prob_arr_test,labels_train,labels_test):
    """
    Compute the probas for data_test for every model in basic model.
    
    input :
        advced_model = list of (clf,"clf_name") containing classifiers
        proba_array_train = probas from first layer used to fit the models, array like
        labels_train = labels associated to data_train, np.array
        prob_arr_test = probas from first layer used to test the models, array like
        labels_train = labels associated to data_test, np.array

    output :
        best_score = best score from all the models, float
        best_model = Best score and model name, string
    """
    nb_advced = len(advced_model)
    score_list = []
    for i in xrange(nb_advced):
        model = advced_model[i][0]
        model_name = advced_model[i][1]
        model.fit(proba_array_train,labels_train)
        score = met.accuracy_score(labels_test,model.predict(prob_arr_test))
        score_list.append(score)
        print model_name+" acc score:", score
    best_score = max(score_list)
    best_model = advced_model[np.argmax(score_list)][1]
    print "\n Score: ",best_score," with ",best_model
    return best_score, best_model
        
def nb_of_token(data,token):
    points= []
    for comment in data: 
        l = sum(1 for c in comment if c==token)
        points.append(l)
    points = np.array(points)
    return points
    
def grade_check(data):
    n = len(data)    
    matr = np.zeros((n,11))
    for l in xrange(n):
        for i in range(11):
            if((str(i)+"/10"in data[l])or(str(i)+"out of 10"in data[l])):
                matr[l,i] = 1
    return matr

def new_feat(data):
    
    #l contains the length of each review
    l = ct.computelength(data)
    l = np.reshape(l,(len(data),))
    l = (l-np.mean(l))#/np.var(l)
    
    #Count number of upper_case
    up = ct.computeUpperCase(data)
    up = np.reshape(up,(len(data),))
    up = (up-np.mean(up))#/np.var(up)
    
    # Nb of point
    poi = nb_of_token(data,".")
    poi = (poi-np.mean(poi))
    
    # Nb of exlamation point
    poi2 = nb_of_token(data,"!")
    poi2 = (poi2-np.mean(poi2))
    
    # Nb of ? point
    poi3 = nb_of_token(data,"?")
    poi3 = (poi3-np.mean(poi3))
   
    # New feature Matrix
    matr = grade_check(data)    
    new_mat = np.zeros((25000,16))
    new_mat[:,0] = np.reshape(l,(25000,))
    new_mat[:,1] = np.reshape(up,(25000,))
    new_mat[:,2] = poi
    new_mat[:,3] = poi2
    new_mat[:,4] = poi3
    new_mat[:,5:] = matr

    return new_mat