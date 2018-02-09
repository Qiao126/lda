from lda_model_train import iter_ap, dump_dir2
from collections import Counter, OrderedDict
from lda_model_train import model_file1, model_file2, model_file3
from lda_model_train import dic_file1, dic_file2, dic_file3
from lda_model_train import test_file3
from operator import itemgetter
import gensim
import numpy as np
import json
import random

SEED = 126


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])

def map(model_file, dic_file, test, i):
    test_size = len(test.keys())
    dictionary = gensim.corpora.Dictionary().load(dic_file)
    #Knum = np.arange(10, 160, 10) # number of topics
    Knum = [100]
    for K in Knum:
        rs = []
        doc_topics = {}
        sim_dist = {}
        print("Load model: " + str(model_file) + str(K) + "---------------------------------------", i)
        lda = gensim.models.ldamodel.LdaModel.load(model_file + str(K) + '.' + str(i))
        for did, doc in test.items():
            topics = lda.get_document_topics(dictionary.doc2bow(doc[1]))  #list of (topic-id, prob)
            #topics = sorted(topics, key=itemgetter(1), reverse=True) # order by desc prob
            #topics = topics[:int(K/5)]  #get top topics
            doc_topics[did] = topics
        docs = doc_topics.keys()
        #print(docs)
        for (i, did) in enumerate(docs):
            #print("topic K: ", K, "doc:", i)
            if docs[i] not in sim_dist:
                sim_dist[docs[i]] = {}
            max_sim = 0
            for j in range(i+1, test_size):
                if docs[j] not in sim_dist:
                    sim_dist[docs[j]] = {}
                sim = gensim.matutils.cossim(doc_topics[docs[i]], doc_topics[docs[j]]) #match topic-id separately in vec1, vec2 to calculate cosine
                sim_dist[docs[i]][docs[j]] = sim
                sim_dist[docs[j]][docs[i]] = sim #save twice
        rs = []
        for (i, did) in enumerate(docs):
            r = []
            #print("doc", did, "-------------------------------------------")
            rank_dist = OrderedDict(sorted(sim_dist[did].items(), key=itemgetter(1), reverse=True))
            for (key, value) in rank_dist.items():
                #print(key, value)
                if test[did][0] == test[key][0]: #two docs have the same tag
                    r.append(1)
                else:
                    r.append(0)
            rs.append(r)
        print(mean_average_precision(rs))


def main():
    """
    #with open("vocalbulary.txt", 'r') as data_file:
    #    vol  = json.load(data_file)['vocalbulary']
    vol = None

    test = {}
    ids = []
    tags = []
    test_doc_list = []
    i = 0
    for did, tag, tokens in iter_ap(dump_dir2, "test", vol):
        if tag != 'unknown': #tag:
            ids.append(did)
            tags.append(tag)#(tag[0])
            test_doc_list.append(list(tokens))
            idstr = str(did)
            print(idstr, tag)#tag[0])
            test[idstr] = (tag, list(tokens))#(tag[0], list(tokens))
        #if len(test_doc_list) == test_size:   # size of test set
        #    print(i)
        #    break
        i += 1
    print("#docs: ", i)
    print(len(ids), len(tags), len(test_doc_list))
    print(Counter(tags))

    random.seed(SEED)
    random.shuffle(ids)

    random.seed(SEED)
    random.shuffle(tags)

    random.seed(SEED)
    random.shuffle(test_doc_list)

    test_size = len(ids) / 10
    print("test_size:", test_size)

    with open(test_file3, 'w') as outfile:
        json.dump({"test_size": test_size,
                    "test_doc_tag" : tags,
                    "test_doc_list": test_doc_list,
                    "test_doc_id" : ids}, outfile)
    """
    with open(test_file3, 'r') as data_file:
        data  = json.load(data_file)
        test_size = data['test_size']
        #test_size = 1000
        test_doc_tag = data['test_doc_tag'][:test_size]
        test_doc_list = data['test_doc_list'][:test_size]
        test_doc_id = data['test_doc_id'][:test_size]
    #print(len(set(test_doc_id).intersection(ids))) # check whether it's the same test set
    print(Counter(test_doc_tag))
    test = {}
    for did, tag, doc in zip(test_doc_id, test_doc_tag, test_doc_list):
        test[did] = (tag, doc)

    """ Check overlapping between MAP vocalbulary and input vocalbulary """
    """
    map_vol = gensim.corpora.Dictionary(test_doc_list).values()
    #vol1 = gensim.corpora.Dictionary().load(dic_file1).values()
    vol2 = gensim.corpora.Dictionary().load(dic_file2).values()
    #print(len(vol1), len(vol2))
    #print(len(set(vol1).intersection(vol2)))
    #vol3 = gensim.corpora.Dictionary().load(dic_file3).values()
    #print(len(vol1), len(vol2), len(vol3), len(map_vol))
    #print("% from wiki:", len(set(vol1).intersection(map_vol)), len(set(vol1).intersection(map_vol))/float(len(vol1)) )
    print("% from ap:", len(set(vol2).intersection(map_vol)), len(set(vol2).intersection(map_vol))/float(len(vol2)) )
    #print("% from merged:", len(set(vol2).intersection(map_vol)), len(set(vol2).intersection(map_vol))/float(len(vol3)) )

    #with open("vocalbulary.txt", 'w') as outfile:
    #    json.dump({"vocalbulary" : list(set(vol2).intersection(vol1))}, outfile)

    """
    for i in range(20):  #20 folds
        #map(model_file1, dic_file1, test)
        #map(model_file2, dic_file2, test)
        map(model_file2, 'lda.ap2.' + str(i) + '.dictionary', test, i)
        #map(model_file3, dic_file3, test)


if __name__ == '__main__':
    main()
