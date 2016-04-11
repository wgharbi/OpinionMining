import networkx as nx
import string
#from sys import maxint
import pandas as pd
import numpy as np
import time
import re
import os.path
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
#num_documents: number of documents
#clean_train_documents: the collection
#unique_words: list of all the words we found 
#sliding_window: window size
#train_par: if true we are in the training documents
#idf_learned
def createGraphFeatures(num_documents,clean_train_documents,unique_words,sliding_window,train_par,idf_learned):
    #features = np.zeros((num_documents,len(unique_words)))#where we are going to put the features
    features = sp.lil_matrix((num_documents,len(unique_words)))
    unique_words_len = len(unique_words)
    term_num_docs = {} #dictionay of each word with a count of that word through out the collections
    idf_col = {}#dictionay of each word with the idf of that word tfidf = TfidfVectorizer()
    
    #TO DO:    
    #1.idf_col:IDF for the collection
    #    if in training phase compute it
    #    else use the one provided
    
    #2. term_num_docs : count of the words in the collection
    #    if in training phase populate it
    #    else use the one provided
    if train_par:
        tfidf = TfidfVectorizer(ngram_range=(1,2), binary=False,analyzer = "word",lowercase= True,norm=None)
        tfidf.fit(clean_train_documents)
        idf_col = { word : tfidf.idf_[k] for word,k in tfidf.vocabulary_.items()}
#        #for all documents
#        for i in range( 0,num_documents ):
#            #count word occurrences through the collection (for idf) put the count in term_num_docs
#            if len(wordList2)>1:
#                countWords(wordList2,term_num_docs) #TODO: implement this function 
#            #TODO: calculate the idf for all words
            
        # for the testing set
    else:
        #use the existing ones if we are in the test data
        idf_col = idf_learned 
        term_num_docs=unique_words
        
    print("Creating the graph of words for each document...")    
    totalNodes = 0
    totalEdges = 0
    
    #go over all documents
    for i in range( 0,num_documents ):
        print "converting to graph document number %d"%i
        wordList1 = clean_train_documents[i].split(None)
        #wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
        wordList2 = [x.lower() for x in wordList1]
        docLen = len(wordList2)
        print docLen
        #the graph
        dG = nx.Graph()

        if len(wordList2)>1:
            populateGraph(wordList2,dG,sliding_window)
            dG.remove_edges_from(dG.selfloop_edges())
            centrality = nx.degree_centrality(dG) #dictionary of centralities (node:degree)

            totalNodes += dG.number_of_nodes()
            totalEdges += dG.number_of_edges()
            
            #TODO : implement comments bellow
            # for all nodes
                #If they are in the desired features
                    #compute the TW-IDF score and put it in features[i,unique_words.index(g)]
            for k,node_term in enumerate(dG.nodes()):
                if node_term in idf_col:
                    features[(i,unique_words[node_term])] = centrality[node_term]*idf_col[node_term]
    if train_par:
        nodes_ret=tfidf.vocabulary_
#        print("Percentage of features kept:"+str(feature_reduction))
#        print("Average number of nodes:"+str(float(totalNodes)/num_documents))
#        print("Average number of edges:"+str(float(totalEdges)/num_documents))
    else:
        nodes_ret=term_num_docs
    #return 1: features, 2: idf values (for the test data), 3: the list of terms 
    Newfeatures = features.tocsr()
    return Newfeatures, idf_col, nodes_ret
    
    
def populateGraph(wordList,dG,sliding_window):
    #TODO: implement this function
    #For each position/word in the word list:
        #add the -new- word in the graph
        #for all words -forward- within the window size
            #add new words as new nodes 
            #add edges among all word within the window
    for k,word in enumerate(wordList):
        if not dG.has_node(word):
            dG.add_node(word)
        tempW = sliding_window
        if k+sliding_window - len(wordList)>0:
            tempW  = len(wordList)-k
        for j in range(1,tempW):
            next_word = wordList[k+j]
            dG.add_edge(word, next_word)
 
def countWords(wordList,term_num_docs):
    #TODO: implement this function
    #add the terms from the wordlist to the term_num_docs dictionary or increase its count
    for k,w in enumerate(wordList):
        break
    return