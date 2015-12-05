import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



from time import time

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
import cleantext as ct
import loadFiles as lf



#%% Pre-process the data
data, labels = lf.loadLabeled("./train")
#l contains the length of each review
l = ct.computelength(data)

#Remove html tags
train = ct.removehtml(data)

#Create the dictionnary
data=ct.stemTokenize(train)  

#Compute tf-idf
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data)

#tfidf_matrix = tfidf_matrix.toarray()
print "size of the matrix : ", tfidf_matrix.shape



#%% Start training parameters

#define test set and traing set
data_train, data_test, labels_train, labels_test = train_test_split(tfidf_matrix, labels, test_size = 0.4, random_state  =42)
#Fix the number of models to train 
nb_models = 2

labels_predicted = np.zeros((len(labels_test),nb_models))


#%% Fit first model : Multinomial Naive Bayes Model

from sklearn.naive_bayes import MultinomialNB
t0 = time()
clf = MultinomialNB(alpha = .01)
y_score = clf.fit(data_train, labels_train)
labels_predicted[:,0]= clf.predict(data_test)
t1=time() -t0
print "-------------------Vectorizing and fitting the model took %s"%t1,"sec---------------"
print ""
print "classification report :"
print classification_report(labels_test, labels_predicted[:,0])
print "the accuracy score is :", accuracy_score(labels_test, labels_predicted[:,0])



#%% Fit second model : SVC

from sklearn.svm import LinearSVC
t1 = time()
clf2 = LinearSVC(C=1.0)
y_score2 = clf2.fit(data_train, labels_train)
labels_predicted[:,1]= clf2.predict(data_test)
t2=time() -t1
print "-------------------Vectorizing and fitting the linear SVC took %s"%t2,"sec---------------"
print "classification report"
print classification_report(labels_test, labels_predicted[:,1])

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


#%%Compute the ROC curve of all models learnt
"""
compute the roc curve and the area under curve 

"""
from sklearn.metrics import roc_curve, auc

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(nb_models) :
    fpr[i], tpr[i], _ = roc_curve(labels_test, labels_predicted[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print "area under the ROC Curve", roc_auc

for i in range(nb_models) :
    plt.plot(fpr[i], tpr[i], label = 'ROC curve model %s'%i)
    
plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel( 'True Positive Rate')

plt.title('Models ROC curve')
plt.legend(loc ='lower right')
plt.show()
