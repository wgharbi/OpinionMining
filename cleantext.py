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
