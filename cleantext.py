#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#                               Preprocessing Functions                       #
#                                                                             #
###############################################################################

from bs4 import BeautifulSoup
import unicodedata   
import numpy as np


#the following function removes the HTML tags from the comments
def removehtml(data):
	newdata= []
	for comment in data:

		temp = BeautifulSoup(comment)
		temp2 = temp.get_text()
		comment = unicodedata.normalize('NFKD', temp2).encode('ascii','ignore')
		newdata.append(comment)
	return newdata

def computelength(data):
    length = []
    for comment in data: 
        l = len(comment)
        length.append(l)
    length = np.array(length)
    length = np.expand_dims(length,axis=1)
    return length

def computeUpperCase(data):
    uppercase= []
    for comment in data: 
        l = sum(1 for c in comment if c.isupper())
        uppercase.append(l)
    uppercase = np.array(uppercase)
    uppercase = np.expand_dims(uppercase,axis=1)
    return uppercase

def countexclamation(data):
    exclamation = []
    for comment in data: 
        l = sum(1 for c in comment if c =="!")
        exclamation.append(l)
    exclamation = np.array(exclamation)
    exclamation = np.expand_dims(exclamation,axis=1)
    return exclamation

def countquestionmark(data):
    questionmark  = []
    for comment in data: 
        l = sum(1 for c in comment if c =="?")
        questionmark.append(l)
    questionmark = np.array(questionmark)
    questionmark = np.expand_dims(questionmark,axis=1)
    return questionmark




#Create a stemmed tokenized - dictionnary from a corpus
def stemTokenize(data):
    from nltk.stem.porter import PorterStemmer
    from nltk import word_tokenize  
    import string
    from nltk.corpus import stopwords
    punctuation = list(set(string.punctuation))
    for a in ["!", "?", "/"]:
        punctuation.remove(a)
    stemmer = PorterStemmer()
    data2=[]
    stop = stopwords.words('english')
    
    for review in data: 
        review = "".join([w for w in review.lower() if (w not in punctuation)])
        words = word_tokenize(review)
        #word_list =[]
        #for word in words: 
        #    if word not in stop:
        #        word=stemmer.stem(word)
        #        word_list.append(word)

        #review = " ".join(word_list)
        data2.append(review)

    return data2
    
    
#Calculates the weights transformation for NBSVM
def nbsvmMatrix(data,labels,alpha):
    import scipy.sparse as sp
    import numpy as np
    from sklearn.preprocessing import binarize
    
    nb_doc, voc_length = data.shape
    pos_idx=[labels==1][0].astype(int)
    neg_idx=[labels==0][0].astype(int)
    #Store the indicator vectors in sparse format to accelerate the computations
    pos_idx=sp.csr_matrix(pos_idx.T)
    neg_idx=sp.csr_matrix(neg_idx)
    #Use sparse format dot product to get a weightning vector stored in sparse format
    alpha_vec=sp.csr_matrix(alpha*np.ones(voc_length))
    p = (alpha_vec + pos_idx.dot(data)) 
    norm_p = p.sum()
    p = p.multiply(1/norm_p)
    print p.toarray()
    q = (alpha_vec + neg_idx.dot(data))
    norm_q = q.sum()
    q = q.multiply(1/norm_q)
    print q.toarray()
    # j'ai enlevé la fonction power
    ratio = sp.csr_matrix(np.log((p.multiply(sp.csr_matrix(np.expand_dims(q.toarray()[0]**(-1),axis=0)))).data))
    print ratio.toarray()    
    #We need now to recompute "f", our binarized word counter
    f_hat = binarize(data, threshold = 0.0) #Binarized to enhance classification
    
    f_tilde = f_hat.multiply(ratio)
    
    return f_tilde
    
#Calculates 
    

    
    
    
    