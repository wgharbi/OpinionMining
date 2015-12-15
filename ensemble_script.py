# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:00:52 2015

@author: ivaylo
"""

from functions import *

############################# Parameters ######################################

# Loading of data
all_train_data, Class = loadLabeled('./train')
# All the models
MnBayes = MNB(alpha=0.1, class_prior=None, fit_prior=True)
Log_reg = LR(C=30,penalty = 'l2', dual = True, random_state = 0)
gbc = GradientBoostingClassifier(n_estimators=200)
rf = RandomForestClassifier(n_estimators=5)
stoch_grad = SGD(loss = 'modified_huber')
svc = SVC(C=100)
ada = AdaBoostClassifier(n_estimators=5)
# First Layer Models for TF-IDF
basic_model1 = [(MnBayes,"Bayes"),(Log_reg,"LogReg"),(stoch_grad,"SGD")]
# First Layer Models for New features
basic_model2 = [(gbc,"GradientBoosting")]
# Second Layer Models
advced_model = [(gbc,"GradientBoosting"),(rf,"RandomForest"),(svc,"svc"),(ada,"AdaBoost")]
# Chi2 on tf_idf 
k_best_nb = 400000
# ngram_range
ngram_rg = (1,2)
# nb of component to be taken for PCA/ICA/CHI2
n_component_list = range(1,4)

############################# Script ##########################################



# New features Matrix
new_mat = new_feat(all_train_data)

# Standard cleaning and TF-IDF Matrix
all_train_cleaned = []
for i in xrange(len(all_train_data)):
    all_train_cleaned.append(" ".join(review_to_wordlist(all_train_data[i])))

tfv = TFIV(ngram_range=ngram_rg,stop_words=None,binary=False)
tfv.fit(all_train_cleaned)
all_train_cleaned = tfv.transform(all_train_cleaned)

# Split Train/Test
data_train, data_test, labels_train, labels_test, new_mat_train, new_mat_test = train_test_split(all_train_cleaned, Class, new_mat, test_size = 0.4, random_state  =42)

# Feature Selection
chi = SelectKBest(chi2,k=k_best_nb)
data_train = chi.fit_transform(data_train,labels_train)
data_test = chi.transform(data_test)
# First Layer Models for TF-IDF
proba1, basic_score1, basic_name1 = first_layer(basic_model1, data_train, labels_train,data_test,labels_test)
# First Layer Models for New features
proba2,basic_score2, basic_name2 = first_layer(basic_model2, new_mat_train, labels_train,new_mat_test,labels_test)
# Grouping the first layer probas
proba = np.hstack([proba1,proba2])


# Best model and score
best_score_list = []
best_model_list = []

# Loop to check the results
#for n_comp in n_component_list:

# Chi2/PCA/ICA
n_comp = 2
ica = FastICA(n_components=n_comp)
pca = PCA(n_components=n_comp)
chi = SelectKBest(chi2,k=n_comp)

proba_chi = chi.fit_transform(proba,labels_test)
proba_ica = ica.fit_transform(proba)
proba_pca = pca.fit_transform(proba)
# Final Model
prob_l = [(proba,"normal"),(proba_chi,"chi2"),(proba_ica,"ica"),(proba_pca,"pca")]
for prob in prob_l:
    if(prob[1]=="normal"):
        print "\n",prob[1],"\n"
    else:
        print "\n",prob[1],str(n_comp),"component","\n"
        
    # New split for the 2nd Layer
    prob_trai, prob_test, lab_tr, lab_te = train_test_split(prob[0],labels_test,test_size = 0.4, random_state  =42)
    # Second Layer
    best_score, best_model = second_layer(advced_model, prob_trai,prob_test, lab_tr,lab_te)
    best_score_list.append(best_score)        
    best_model_list.append("on "+str(n_comp)+" component(s) "+prob[1]+" "+best_model)

print "\n Basic score and Model: ", basic_score1, basic_name1
print "\n Best score and Model: ", max(best_score_list), best_model_list[np.argmax(best_score_list)]

#%%

# Plots for the reports
#
#ica = FastICA(n_components=2)
#pca = PCA(n_components=2)
#chi = SelectKBest(chi2,k=2)
#
#proba_chi = chi.fit_transform(proba,labels_test)
#proba_ica = ica.fit_transform(proba)
#proba_pca = pca.fit_transform(proba)
#
#pl.figure()
#pl.title("chi2 w/ features engineering")
#pl.plot(proba_chi[:,0][labels_test==0],proba_chi[:,1][labels_test==0],'b+',label="0")
#pl.plot(proba_chi[:,0][labels_test==1],proba_chi[:,1][labels_test==1],'r+',label="1")
#pl.legend(loc=0)
#
#pl.figure()
#pl.title("ica w/ features engineering")
#pl.plot(proba_ica[:,0][labels_test==0],proba_ica[:,1][labels_test==0],'b+',label="0")
#pl.plot(proba_ica[:,0][labels_test==1],proba_ica[:,1][labels_test==1],'r+',label="1")
#pl.legend(loc=0)
#
#pl.figure()
#pl.title("pca /w features engineering")
#pl.plot(proba_pca[:,0][labels_test==0],proba_pca[:,1][labels_test==0],'b+',label="0")
#pl.plot(proba_pca[:,0][labels_test==1],proba_pca[:,1][labels_test==1],'r+',label="1")
#pl.legend(loc=0)
#
#
#ica = FastICA(n_components=2)
#pca = PCA(n_components=2)
#chi = SelectKBest(chi2,k=2)
#
#proba_chi = chi.fit_transform(proba1,labels_test)
#proba_ica = ica.fit_transform(proba1)
#proba_pca = pca.fit_transform(proba1)
#
#pl.figure()
#pl.title("chi2 w/o features engineering")
#pl.plot(proba_chi[:,0][labels_test==0],proba_chi[:,1][labels_test==0],'b+',label="0")
#pl.plot(proba_chi[:,0][labels_test==1],proba_chi[:,1][labels_test==1],'r+',label="1")
#pl.legend(loc=0)
#
#pl.figure()
#pl.title("ica w/o features engineering")
#pl.plot(proba_ica[:,0][labels_test==0],proba_ica[:,1][labels_test==0],'b+',label="0")
#pl.plot(proba_ica[:,0][labels_test==1],proba_ica[:,1][labels_test==1],'r+',label="1")
#pl.legend(loc=0)
#
#pl.figure()
#pl.title("pca w/o features engineering")
#pl.plot(proba_pca[:,0][labels_test==0],proba_pca[:,1][labels_test==0],'b+',label="0")
#pl.plot(proba_pca[:,0][labels_test==1],proba_pca[:,1][labels_test==1],'r+',label="1")
#pl.legend(loc=0)