import pandas as pd
import numpy as np 

#preprocessing librairies and functions
import cleantext as ct
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from models import NBmatrix

#Models librairies
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

#Other
from time import time
import matplotlib.pyplot as plt
import loadFiles as lf
import numpy as np

#%% Pre-process the data
data, labels = lf.loadLabeled("./train")


#l = ct.computelength(data)
#up = ct.computeUpperCase(data)
#exc = countexclamation(data)
#ques = countquestionmark(data)
#X = np.array([l,up,exc,ques])

#X= X_T

#Remove html tags
train = ct.removehtml(data)

#Create the dictionnary (WARNING nltk should be up-to-date)
data=ct.stemTokenize(train)  

#Compute tf-idf including n_grams of size 2 
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), binary=False)

#Compute a count_vectorizer including n_grams of size 2
count_vectorizer = CountVectorizer(ngram_range=(1,2),binary=False)

#Comptute a NB matrix as describe by Wang & Manning
nb_vectorizer = NBmatrix(alpha = 1.0 ,bina = True, n_jobs = 1)

#Fit transform on the data
tfidf_matrix = tfidf_vectorizer.fit_transform(data)
count_matrix = count_vectorizer.fit_transform(data)
nb_matrix = nb_vectorizer.fit_transform(count_matrix,labels)

print "size of the matrix : ", tfidf_matrix.shape
average_nb_words = np.mean(count_matrix.sum(axis=1))
print "Average number of words per review : ", average_nb_words
dic_size = count_matrix.shape[1]
print "dictionnary size : " , dic_size
sparsity = 1-float(count_matrix.nnz)/(25000.0*dic_size)
print "Sparsity of the data : ", sparsity



#%% Start training parameters

#define test set and traing set
data_train, data_test, labels_train, labels_test = train_test_split(tfidf_matrix, labels, test_size = 0.4, random_state  =42)


#We use the same random state so that the split will be the same as on the train/test before
data_train2, data_test2, labels_train2, labels_test2 = train_test_split(nb_matrix, labels, test_size = 0.4, random_state  =42)



#Fix the number of models to train 

model_names=[]
labels_predicted = np.expand_dims(np.zeros(len(labels_test)),axis=1)

#%% Fit models
from models import naiveBayes
model_names.append("MultinomialNB")
prediction1 = np.expand_dims(naiveBayes(data_train,labels_train,data_test,labels_test,show_infos=True),axis=1)
labels_predicted=np.append(labels_predicted, prediction1 ,axis=1)

from models import svc
model_names.append("LinearSVC")
prediction2 = np.expand_dims(svc(data_train,labels_train,data_test,labels_test,C=1,show_infos=True),axis=1)
labels_predicted=np.append(labels_predicted, prediction2 ,axis=1)

from models import logRegression
model_names.append("logRegression")
prediction3 = np.expand_dims(logRegression(data_train,labels_train,data_test,labels_test,show_infos=True),axis=1)
labels_predicted=np.append(labels_predicted, prediction3 ,axis=1)



#%% Fit 4th model : NBSVM
from sklearn.svm import LinearSVC
from models import NBmatrix

model_names.append("NBSVM")

t1 = time()

clf4 = LinearSVC(C=1)
y_score4 = clf4.fit(data_train2, labels_train2)
prediction4 = np.expand_dims(clf4.predict(data_test2),axis=1)
labels_predicted=np.append(labels_predicted, prediction4 ,axis=1)
t2=time() -t1
print "-------------------Vectorizing and fitting the Log-reg took %s"%t2,"sec---------------"
print "classification report"
print classification_report(labels_test2, prediction)
print "the accuracy score is :", accuracy_score(labels_test2, prediction)

labels_predicted=labels_predicted[:,1:] #Remove the first column (initialization)
#%%Compute the ROC curve of all models learnt
"""
compute the roc curve and the area under curve 

"""
from sklearn.metrics import roc_curve, auc

fpr = dict()
tpr = dict()
roc_auc = dict()

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
svc.fit(data_train2, labels_train)
print "best parameter", svc.best_params_
labels_predicted= svc.predict(data_test)
print "classification report"
print classification_report(labels_test, labels_predicted)
from sklearn.metrics import accuracy_score
print "the accuracy score is", accuracy_score(labels_test, labels_predicted)
#the accuracy score is 0.8783 for svc and 0.83 for NB
