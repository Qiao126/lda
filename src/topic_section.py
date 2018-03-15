from lda_model_train import model_file2
from lda_model_train import test_file3
import json
from collections import Counter
import matplotlib.pyplot as pyplot

with open(test_file3, 'r') as data_file:
    data  = json.load(data_file)
    test_size = data['test_size']
    test_doc_tag = data['test_doc_tag'][:test_size]
    test_doc_list = data['test_doc_list'][:test_size]
    test_doc_id = data['test_doc_id'][:test_size]

model_file = model_file2 + '18.66'
lda = gensim.models.ldamodel.LdaModel.load(model_file)

for doc, section in zip(test_doc_list, test_doc_tag):
    topics = model.get_document_topics(dictionary.doc2bow(doc), minimum_probability=1.0/K)  # list of (topic-id, prob)
    num_topics = len(topics)
    topics = sorted(topics, key=itemgetter(1), reverse=True) # order by desc prob
    for t in topics:
        if not topic_section[t]:
            topic_section[t] = []
        else:
            topic_section[t].append(section)

for i, t in enumerate(topic_section.keys()):
    counts = Counter(topic_section[t])
    pyplot.pie([float(v) for v in counts.values()], labels=[float(k) for k in counts], autopct=None)
    pyplot.savefig("pie." + str(i) + '.png')
