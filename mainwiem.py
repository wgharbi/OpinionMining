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

data = data2[:10000]
labels = labels[:10000]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data)

tfidf_matrix = tfidf_matrix.toarray()
print "size of the matrix", tfidf_matrix.shape

data_train, data_test, labels_train, labels_test = train_test_split(tfidf_matrix, labels, test_size = 0.4, random_state  =42) 

