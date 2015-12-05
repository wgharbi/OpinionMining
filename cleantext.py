#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#                               Preprocessing Functions                       #
#                                                                             #
###############################################################################

from bs4 import BeautifulSoup
import unicodedata   


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
	return length

#Create a stemmed tokenized - dictionnary from a corpus
def stemTokenize(data):
    from nltk.stem.porter import PorterStemmer
    from nltk import word_tokenize	
    import string
    from nltk.corpus import stopwords

    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    data2=[]
    stop = stopwords.words('english')
    
    for review in data: 
	review = "".join([w for w in review.lower() if (w not in punctuation)])
	words = word_tokenize(review)
	word_list =[]
	for word in words: 
		if word not in stop:
			word=stemmer.stem(word)
			word_list.append(word)

	review = " ".join(word_list)

	data2.append(review)

    return data2