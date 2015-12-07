import pandas as pd
import numpy as np 

#preprocessing librairies and functions
import cleantext as ct
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#Models librairies
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

#Other
from time import time
import scipy.sparse as sp

import matplotlib.pyplot as plt



import loadFiles as lf



#%% Load the data and compute alternative features
data, labels = lf.loadLabeled("./train")
#l contains the length of each review
l = sp.computelength(data)
l = sp.csr_matrix(l)

#Count number of upper_case
up = ct.computeUpperCase(data)
up = sp.csr_matrix(up)


#Count excalmations
ex = sp.countexclamation(data)
ex = sp.csr_matrix(ex)

#Count question marks
qu = sp.countquestionmark(data)
qu = sp.csr_matrix(qu)


#%%Pre process

#Remove html tags
train = ct.removehtml(data)

#Create the dictionnary
data=ct.stemTokenize(train)  

#Compute tf-idf including n_grams of size 2 
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), binary=False)
#HUGO : j'ai testé avec une tf-idf "binarisée" pour voir si cela permettait une ammélioration du score, ce qui n'est pas le cas

#Compute a count_vectorizer including n_grams of size 2
count_vect = CountVectorizer(ngram_range=(1,2),binary=False)


tfidf_matrix = tfidf_vectorizer.fit_transform(data)
count_matrix = count_vect.fit_transform(data)
nbsvm_matrix = ct.nbsvmMatrix(count_matrix,labels,alpha=1)
#tfidf_matrix = tfidf_matrix.toarray()
print "size of the matrix : ", tfidf_matrix.shape




#%% Start training parameters

#define test set and traing set
data_train, data_test, labels_train, labels_test = train_test_split(tfidf_matrix, labels, test_size = 0.4, random_state  =42)
data_train2, data_test2, labels_train2, labels_test2 = train_test_split(nbsvm_matrix, labels, test_size = 0.4, random_state  =42)

#Fix the number of models to train 

model_names=[]
labels_predicted = np.expand_dims(np.zeros(len(labels_test)),axis=1)

#%% Fit models
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



#%% Fit 4th model : NBSVM
from sklearn.svm import LinearSVC
model_names.append("NBSVM")

#HUGO : la partie "interpolation" grâce au paramètre beta n'est pas encore implémentée
#Du coup ici c'est comme si on entraine avec beta=1

t1 = time()
clf4 = LinearSVC(C=1)
y_score4 = clf4.fit(data_train2, labels_train2)
prediction = np.expand_dims(clf4.predict(data_test2),axis=1)
labels_predicted=np.append(labels_predicted, prediction ,axis=1)
t2=time() -t1
print "-------------------Vectorizing and fitting the Log-reg took %s"%t2,"sec---------------"
print "classification report"
print classification_report(labels_test2, prediction)
print "the accuracy score is :", accuracy_score(labels_test2, prediction)


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

#%% Find best parameters for SVC
from sklearn.grid_search import GridSearchCV
svc = GridSearchCV( LinearSVC(),  cv = 5, param_grid={"C":np.logspace(-2, 2, 5)})
svc.fit(data_train, labels_train)
print "best parameter", svc.best_params_
labels_predicted= svc.predict(data_test)
print "classification report"
print classification_report(labels_test, labels_predicted)
from sklearn.metrics import accuracy_score
print "the accuracy score is", accuracy_score(labels_test, labels_predicted)
#the accuracy score is 0.8783 for svc and 0.83 for NB
