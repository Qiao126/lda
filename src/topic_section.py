from lda_model_train import model_file2
from lda_model_train import test_file3
import json
from collections import Counter, OrderedDict
import matplotlib.pyplot as pyplot
import gensim
from operator import itemgetter
import numpy as np

with open(test_file3, 'r') as data_file:
    data  = json.load(data_file)
    test_size = data['test_size']
    test_doc_tag = data['test_doc_tag'][:test_size]
    test_doc_list = data['test_doc_list'][:test_size]
    test_doc_id = data['test_doc_id'][:test_size]

K = 18
i = 66
c = 18 #sections
model_file = model_file2 + str(K) + '.' + str(i)
lda = gensim.models.ldamodel.LdaModel.load(model_file)
dic_file = 'lda.ap2.dictionary' #'lda.ap2.' + str(i) + '.dictionary' #'lda.ap2.dictionary'
dictionary2 = gensim.corpora.Dictionary().load(dic_file)

topic_section = {}
for doc, section in zip(test_doc_list, test_doc_tag):
    topics = lda.get_document_topics(dictionary2.doc2bow(doc), minimum_probability=1.0/K)  # list of (topic-id, prob)
    num_topics = len(topics)
    topics = sorted(topics, key=itemgetter(1), reverse=True) # order by desc prob
    for t, _ in topics:
        if not t in topic_section:
            topic_section[t] = []
        else:
            topic_section[t].append(section)

sections = OrderedDict(sorted(Counter(test_doc_tag).items()))

dataMatrix = [[float(v)/sections.values()[idx] for idx, v in enumerate(OrderedDict(sorted(Counter(topic_section[i]).items())).values())] for i in topic_section.keys()]
dataMatrix = [a/np.sum(a) for a in dataMatrix]

a = np.zeros(shape=(K,c))
for i in range(len(dataMatrix)):
    for j in range(len(dataMatrix[i])):
        a[i,j] = dataMatrix[i][j]

print(a.shape)

pyplot.figure()
pyplot.pcolor(a, cmap='hot')
pyplot.savefig("heatmap." + str(K) + ".png")
"""
for i, t in enumerate(topic_section.keys()):
    counts = Counter(topic_section[t])
    pyplot.figure()
    pyplot.pie([float(v) for v in counts.values()], labels=[k for k in counts], autopct=None)
    pyplot.savefig("pie." + str(i) + '.png')
"""
