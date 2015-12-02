import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



from time import time

import matplotlib.pyplot as plt

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

#%% Create a stemmed tokenized - dictionnary and compute the TF-IDF matrix

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


t0 = time()
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data)

#tfidf_matrix = tfidf_matrix.toarray()
print "size of the matrix : ", tfidf_matrix.shape


#%% Start training parameters

#define test set and traing set
data_train, data_test, labels_train, labels_test = train_test_split(tfidf_matrix, labels, test_size = 0.4, random_state  =42)
#Fix the number of models to train 
nb_models = 1

labels_predicted = np.zeros((len(labels_test),nb_models))


#%% Fit first model : Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha = .01)
y_score = clf.fit(data_train, labels_train)
labels_predicted[:,0]= clf.predict(data_test)
t1=time() -t0
print "-------------------Vectorizing and fitting the model took %s"%t1,"sec---------------"
print ""
print "classification report :"
print classification_report(labels_test, labels_predicted[:,0])
print "the accuracy score is :", accuracy_score(labels_test, labels_predicted[:,0])



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
plt.clf()
plt.plot(fpr, tpr, label = 'ROC curve')
plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel( 'True Positive Rate')

plt.title('Naive Bayes model ROC curve')
plt.legend(loc ='lower right')
plt.show()

np.argsort(clf.coef_[i])[-10:]