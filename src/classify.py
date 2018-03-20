from lda_model_train import model_file2
from lda_model_train import test_file4
import json
from collections import Counter, OrderedDict
import gensim
from operator import itemgetter
import numpy as np

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_recall_fscore_support,
    average_precision_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from learn import plot_learning_curve
import re
import os

with open(test_file4, 'r') as data_file:
    data  = json.load(data_file)
    test_size = data['test_size']
    test_doc_tag = data['test_doc_tag'][:test_size]
    test_doc_list = data['test_doc_list'][:test_size]
    test_doc_id = data['test_doc_id'][:test_size]

sections = OrderedDict(sorted(Counter(test_doc_tag).items()))

K = 18
i = 88
c = 18 #sections
model_file = model_file2 + str(K) + '.' + str(i)
lda = gensim.models.ldamodel.LdaModel.load(model_file)
dic_file = 'lda.ap2.dictionary' #'lda.ap2.' + str(i) + '.dictionary' #'lda.ap2.dictionary'
dictionary2 = gensim.corpora.Dictionary().load(dic_file)

X = []
Y = []
topic_section = {}
for did, doc, section in zip(test_doc_id, test_doc_list, test_doc_tag):
    topics = lda.get_document_topics(dictionary2.doc2bow(doc), minimum_probability=0)  # list of (topic-id, prob)
    num_topics = len(topics)
    print(num_topics)
    vec = [x[1] for x in topics]
    X.append(vec)
    Y.append(section)
    doc_hash[did] = topics, section
"""
num = len(X)
with open("train." + str(K) + "." + str(i) + ".txt", 'w') as outfile:
    json.dump({"X": X, "Y": Y}, outfile)
"""
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.asarray(X)

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = np.asarray(Y)

train_num = 0.8 * X.shape[0]
X_train, X_test = X[:train_num], X[train_num:]
Y_train, Y_test = Y[:train_num], Y[train_num:]

rf = RandomForestClassifier(n_estimators=3, max_depth=2, max_features=None)
clf = rf
clf.fit(X_train, y_train)
title = "Learning Curve"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
plot_learning_curve(clf, title, X, Y, cv=cv, n_jobs=4)
plt.show()
