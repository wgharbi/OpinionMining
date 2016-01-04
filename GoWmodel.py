import pandas as pd
import numpy as np 

#preprocessing librairies and functions
import cleantext as ct
from MyGraph import createGraphFeatures
from sklearn.feature_extraction.text import TfidfVectorizer

#Models librairies
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

#Other
import time
import scipy.sparse as sp

import matplotlib.pyplot as plt
import sys
import loadFiles as lf



sliding_window = 2
#%% Load the data and compute alternative features
data, labels = lf.loadLabeled("./train")


#%%Pre process

#Remove html tags
train = ct.removehtml(data)

#Create the dictionnary
data=ct.stemTokenize(train)  
idfs = {}
num_documents = len(data)
print "number of documents %s"%num_documents




tfidf_vect = TfidfVectorizer(ngram_range=(1,2), binary=False,analyzer = "word",lowercase= True,norm=None)
features = tfidf_vect.fit_transform(data)
unique_words = list(set(tfidf_vect.vocabulary_.keys()))
print("Unique words:"+str(len(unique_words)))


   #tw-idf features on train data
#features, idfs_learned, nodes= createGraphFeatures(num_documents,data,tfidf_vect.vocabulary_,sliding_window,True,idfs)
start = time.time()
features, idfs_learned, nodes= createGraphFeatures(num_documents,data,tfidf_vect.vocabulary_,sliding_window,True,idfs)
end = time.time()
print "it took %d" %start-end
print("Total time to build features:\t"+str(end - start))
data_train, data_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state  =42)

model_names=[]
labels_predicted = np.expand_dims(np.zeros(len(labels_test)),axis=1)


from models import naiveBayes
model_names.append("MultinomialNB")
prediction = np.expand_dims(naiveBayes(data_train,labels_train,data_test,labels_test,show_infos=True),axis=1)
labels_predicted=np.append(labels_predicted, prediction ,axis=1)

from models import svc
model_names.append("LinearSVC")
prediction = np.expand_dims(svc(data_train,labels_train,data_test,labels_test,C=1,show_infos=True),axis=1)
labels_predicted=np.append(labels_predicted, prediction ,axis=1)

from models import logRegression
model_names.append("logRegression")
prediction = np.expand_dims(logRegression(data_train,labels_train,data_test,labels_test,show_infos=True),axis=1)
labels_predicted=np.append(labels_predicted, prediction ,axis=1)

#%%Compute the ROC curve of all models learnt
"""
compute the roc curve and the area under curve 

"""
from sklearn.metrics import roc_curve, auc

fpr = dict()
tpr = dict()
roc_auc = dict()
labels_predicted=labels_predicted[:,1:]
nb_models = labels_predicted.shape[1]
for i in range(nb_models) :
    fpr[i], tpr[i], _ = roc_curve(labels_test, labels_predicted[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print "area under the ROC Curves", roc_auc

for i in range(nb_models) :
    name=model_names[i]
    plt.plot(fpr[i], tpr[i], label = name)
    
plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel( 'True Positive Rate')

plt.title('Models ROC curve')
plt.legend(loc ='lower right')
plt.show()