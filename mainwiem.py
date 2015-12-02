import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split
import cleantext as ct
import loadFiles as lf
from nltk import word_tokenize	
import string
from nltk.corpus import stopwords

data, labels = lf.loadLabeled("./train")
#l = cf.computelength(data)
train = ct.removehtml(data)

punctuation = set(string.punctuation)
stemmer = PorterStemmer()
data2=[]
stop = stopwords.words('english')
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

from time import time
t0 = time()
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data)

#tfidf_matrix = tfidf_matrix.toarray()
print "size of the matrix", tfidf_matrix.shape

data_train, data_test, labels_train, labels_test = train_test_split(tfidf_matrix, labels, test_size = 0.4, random_state  =42) 

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha = .01)
y_score = clf.fit(data_train, labels_train)
labels_predicted= clf.predict(data_test)
t1=time() -t0
print "-------------------Vectorizing and fitting the model took %s------------------" %t1
from sklearn.metrics import classification_report
print "classification report"
print classification_report(labels_test, labels_predicted)


from sklearn.metrics import accuracy_score
print "the accuracy score is", accuracy_score(labels_test, labels_predicted)



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

