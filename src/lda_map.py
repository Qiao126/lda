#!/usr/local/bin/python
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
import psycopg2
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot, inner
from numpy.linalg import norm

SEED = 126


def average_precision(r):
    r = np.asarray(r) != 0
    mean_arrays = []
    count = 0
    res = 0

    for k in range(0, r.size):
        if (r[k]):
            count = count + 1
            res += (float(count) / (k + 1))

    return res / count

"""
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
"""

def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])

simmap = {}
def get_cossim(i, j, doc_topics, docs):
    doc_v = str(min([i,j])) + str(max([i, j]))
    if doc_v in simmap:
        return simmap[doc_v]

    sim = gensim.matutils.cossim(doc_topics[docs[i]], doc_topics[docs[j]]) #match topic-id separately in vec1, vec2 to calculate cosine
    #a = doc_topics[docs[i]]
    #b = doc_topics[docs[j]]
    #sim = np.dot(a, b) / (norm(a) * norm(b))
    #a = np.array(a).reshape(1,-1)
    #b = np.array(b).reshape(1,-1)
    #sim = cosine_similarity(a, b)
    simmap[doc_v] = sim[0][0]
    return sim

def map(model_file, dic_file, test, i):
    #conn = psycopg2.connect("host=localhost dbname=postgres user=postgres")
    #cur = conn.cursor()
    #cur.execute("drop table similarity; ")
    #cur.execute(
    #    """
    #    CREATE TABLE similarity(
    #        id text PRIMARY KEY,
    #        cossim float,
    #        rel integer);
    #    """
    #)
    #conn.commit()
    test_size = len(test.keys())
    dictionary = gensim.corpora.Dictionary().load(dic_file)
    #Knum = np.arange(10, 160, 10) # number of topics
    Knum = [100]
    for K in Knum:
        doc_topics = {}
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Load model: " + str(model_file) + str(K) + "---------------------------------------", i)
        lda = gensim.models.ldamodel.LdaModel.load(model_file + str(K) + '.' + str(i))
        for did, doc in test.items():
            topics = lda.get_document_topics(dictionary.doc2bow(doc[1]))  #list of (topic-id, prob)
            #topics =[x[1] for x in sorted(topics, key=itemgetter(0))]
            #print(len(topics))
            #print(topics)
            doc_topics[did] = topics
        docs = doc_topics.keys()
        rs = []
        for (i, did) in enumerate(docs):
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "got new document")
            sim_dist = {}
            for j in range(test_size):
                sim = get_cossim(i, j, doc_topics, docs) #match topic-id separately in vec1, vec2 to calculate cosine
                if test[docs[i]][0] == test[docs[j]][0]: #two docs have the same tag
                    rel = 1
                else:
                    rel = 0
                sim_dist[docs[j]] = (sim, rel)
                #insert_query = "INSERT INTO similarity VALUES ('{}', {}, {})".format(docs[j], sim, rel)
                #cur.execute(insert_query)
            #conn.commit()

            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "got similarities")
            rank_dist = OrderedDict(sorted(sim_dist.items(), key=lambda x: x[1][0], reverse=True)[:100])
            r = [ x[1] for x in rank_dist.values() ]
            #query = "SELECT rel FROM similarity ORDER BY cossim DESC LIMIT 100; "
            #cur.execute(query)
            #r = cur.fetchall()
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "got relevance ranking")
            ap = average_precision(r)
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "got ap")
            rs.append(ap)
            sim_dist.clear()
            #cur.execute("TRUNCATE similarity;")

        print(np.mean(rs))
    #conn.close ()

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
