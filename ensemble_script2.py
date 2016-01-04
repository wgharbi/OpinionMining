# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:22:09 2015

@author: ivaylo
"""

from ensemble_functions import *
from models import NBmatrix2

############################# Parameters ######################################

# Loading of data
all_train_data, Class = loadLabeled('./train')
# All the models
MnBayes = MNB(alpha=0.1, class_prior=None, fit_prior=True)
Log_reg = LR(C=30,penalty = 'l2', dual = True, random_state = 0)
gbc = GradientBoostingClassifier(n_estimators=200)
rf = RandomForestClassifier(n_estimators=5)
stoch_grad = SGD(loss = 'modified_huber')
svc = SVC(C=1)
ada = AdaBoostClassifier(n_estimators=5)
# First Layer Models for TF-IDF
basic_model1 = [(MnBayes,"Bayes")]#,(Log_reg,"LogReg"),(stoch_grad,"SGD")]
# First Layer Models for New features
basic_model2 = [(gbc,"GradientBoosting")]
# First Layer for NBSVM
basic_model3 = [(Log_reg,"LogReg")]
# Second Layer Models
advced_model = [(gbc,"GradientBoosting"),(rf,"RandomForest"),(svc,"svc"),(ada,"AdaBoost")]
# Chi2 on tf_idf 
k_best_nb = 400000
# ngram_rangeprobafi3
ngram_rg = (1,2)
# nb of component to be taken for PCA/ICA/CHI2
n_component_list = range(1,4)
# Test ratios
layer1_ratio = 0.4


#%%############################# Script ##########################################



# New features Matrix
new_mat = new_feat(all_train_data)

# Standard cleaning and TF-IDF Matrix
all_train_cleaned = []
for i in xrange(len(all_train_data)):
    all_train_cleaned.append(" ".join(review_to_wordlist(all_train_data[i])))

# Split Train/Test Layer 1
data_train, data_test, labels_train, labels_test, new_mat_train, new_mat_test = train_test_split(all_train_cleaned, Class, new_mat, test_size = layer1_ratio, random_state  =42)

# Count Vecotrizer for NBmatrix
count_vect = CountVectorizer(ngram_range=(1,2),binary=False)
count_vect.fit(data_train)
count_matrix = count_vect.transform(data_train)
count_test = count_vect.transform(data_test)

# TF-IDF
tfv = TFIV(ngram_range=ngram_rg,stop_words=None,binary=False)
data_train = tfv.fit_transform(data_train)
data_test = tfv.transform(data_test)

# Feature Selection Tfidf
chi = SelectKBest(chi2,k=k_best_nb)
data_train = chi.fit_transform(data_train,labels_train)
data_test = chi.transform(data_test)

# Feature Selection CountVectorizer
chi = SelectKBest(chi2,k=k_best_nb)
chi.fit_transform(count_matrix,labels_train)
count_matrix = chi.transform(count_matrix)
count_test = chi.transform(count_test)


# Nbmatrix
nbmat = NBmatrix2(1.0,bina=True,n_jobs=1)
nbmat.fit(count_matrix,labels_train)
nbm_test = nbmat.transform(count_test)
nbm_data = nbmat.transform(count_matrix)

########################### Train part ########################################

# First Layer Models for TF-IDF
proba1, basic_score1, basic_name1 = first_layer(basic_model1, data_train, labels_train,data_train,labels_train)
# First Layer Models for New features
proba2,basic_score2, basic_name2 = first_layer(basic_model2, new_mat_train, labels_train,new_mat_train,labels_train)
# First Layer Nbmatrix
proba3, basic_score3, basic_name3 = first_layer(basic_model3, nbm_data, labels_train,nbm_data,labels_train)
# Grouping the first layer probas
proba = np.hstack([proba1,proba2,proba3])

############################## Test part ######################################

# First Layer Models for TF-IDF
probafi1, basic_scorefi1, basic_namefi1 = first_layer(basic_model1, data_train, labels_train,data_test,labels_test)
# First Layer Models for New features
probafi2,basic_scorefi2, basic_namefi2 = first_layer(basic_model2, new_mat_train, labels_train,new_mat_test,labels_test)
# First Layer nbsvm
probafi3, basic_scorefi3, basic_namefi3 = first_layer(basic_model3, nbm_data, labels_train,nbm_test,labels_test)
# Grouping the first layer probas
probafi = np.hstack([probafi1,probafi2,probafi3])

# Chi2/PCA/ICA
n_comp = 2
ica = FastICA(n_components=n_comp)
pca = PCA(n_components=n_comp)
chi = SelectKBest(chi2,k=n_comp)

proba_pca_train = pca.fit_transform(proba)
proba_pca_test = pca.transform(probafi)

# Basic Model score
print " Basic Model Score: ", basic_scorefi1

# Score Final
best_score, best_model = second_layer(advced_model, proba_pca_train, proba_pca_test, labels_train,labels_test)



