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
    
    

    

    

    
    
    
    