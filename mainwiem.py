import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split
import cleantext as ct
import loadFiles as lf
from nltk import word_tokenize	
import string
from nltk.corpus import stopwords

#%% Pre-process the data
data, labels = lf.loadLabeled("./train")
#l contains the length of each review
l = ct.computelength(data)
train = ct.removehtml(data)

punctuation = set(string.punctuation)
stemmer = PorterStemmer()
data2=[]
stop = stopwords.words('english')

#%% Create a stemmed tokenized - dictionnary
for review in train: 
	review = "".join([w for w in review.lower() if (w not in punctuation)])
	words = word_tokenize(review)
	word_list =[]
	for word in words: 
		if word not in stop:
			word=stemmer.stem(word)
			word_list.append(word)

	review = " ".join(word_list)

	data2.append(review)

data = data2
labels = labels

#%% Fit a Naive Bayes Model

from time import time
t0 = time()
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data)

#tfidf_matrix = tfidf_matrix.toarray()
print "size of the matrix", tfidf_matrix.shape

data_train, data_test, labels_train, labels_test = train_test_split(tfidf_matrix, labels, test_size = 0.4, random_state  =42) 
###########################################################################################
#Trying a Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha = .01)
y_score = clf.fit(data_train, labels_train)
labels_predicted= clf.predict(data_test)
t1=time() -t0
print "-------------------Vectorizing and fitting the MNB took %s------------------" %t1
from sklearn.metrics import classification_report
print "classification report"
print classification_report(labels_test, labels_predicted)
from sklearn.metrics import accuracy_score
print "the accuracy score is", accuracy_score(labels_test, labels_predicted)
############################################################################################
#Trying a linear SVC
from sklearn.svm import LinearSVC
clf = LinearSVC(C=1.0)
y_score = clf.fit(data_train, labels_train)
labels_predicted= clf.predict(data_test)
t2=time() -t1
print "-------------------Vectorizing and fitting the linear SVC took %s------------------" %t1
print "classification report"
print classification_report(labels_test, labels_predicted)

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

"""
compute the roc curve and the area under curve 

"""
from sklearn.metrics import roc_curve, auc


fpr, tpr, thresholds = roc_curve(labels_test, y_score[:,1])

roc_auc = auc(fpr, tpr)

print "area under the ROC Curve", roc_auc
import matplotlib.pyplot as plt
plt.clf()
plt.plot(fpr, tpr, label = 'ROC curve')
plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel( 'True Positive Rate')

plt.title(' Receiver operating characteristic example')
plt.legend(loc ='lower right')
plt.show()

np.argsort(clf.coef_[i])[-10:]