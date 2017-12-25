from lda_model_train import iter_ap, dump_dir2
from collections import Counter, OrderedDict
from lda_model_train import model_file1, model_file2, model_file3
from lda_model_train import dic_file1, dic_file2, dic_file3
from operator import itemgetter
import gensim
import numpy as np
import json

test_size = 1000
test_file3 = '../data/test_doc_list_apTag.json'


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
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])

def map(model_file, dic_file, dist):
    dictionary = gensim.corpora.Dictionary().load(dic_file)

    Knum = [10, 50, 100, 500]
    for K in Knum:
        rs = []
        doc_topics = {}
        sim_dist = {}
        print("Load model: " + str(model_file) + str(K) + "---------------------------------------\n")
        lda = gensim.models.ldamodel.LdaModel.load(model_file + str(K))
        for did, doc in dist.items():
            topics = lda.get_document_topics(dictionary.doc2bow(doc[1]))  #list of (topic-id, prob)
            topics = sorted(topics, key=itemgetter(1), reverse=True) # order by desc prob
            topics = topics[:int(K/5)]
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
                sim = gensim.matutils.cossim(doc_topics[docs[i]], doc_topics[docs[j]])
                sim_dist[docs[i]][docs[j]] = sim
                sim_dist[docs[j]][docs[i]] = sim #save twice
        rs = []
        for (i, did) in enumerate(docs):
            r = []
            #print("doc", did, "-------------------------------------------")
            rank_dist = OrderedDict(sorted(sim_dist[did].items(), key=itemgetter(1), reverse=True))
            for (key, value) in rank_dist.items():
                #print(key, value)
                if dist[did][0] == dist[key][0]: #two docs have the same tag
                    r.append(1)
                else:
                    r.append(0)
            rs.append(r)
        print(mean_average_precision(rs))


def main():

    dist = {}
    ids = []
    tags = []
    doc_list = []
    i = 0
    for did, tag, tokens in iter_ap(dump_dir2):
        if tag:
            ids.append(did)
            tags.append(tag[0])
            doc_list.append(list(tokens))
            idstr = str(did)
            print(idstr, tag[0])
            dist[idstr] = (tag[0], list(tokens))
        if len(doc_list) == test_size:   # size of test set
            print(i)
            break
        i += 1

    #print(len(ids), len(tags), len(doc_list))
    #print(ids)
    print(Counter(tags))

    with open(test_file3, 'w') as outfile:
        json.dump({"test_doc_list" : doc_list}, outfile)

    with open(test_file3, 'r') as data_file:
        test_docs_list  = json.load(data_file)['test_doc_list']
    test_docs3 = test_docs_list

    """ Check overlapping between MAP vocalbulary and input vocalbulary """
    map_vol = gensim.corpora.Dictionary(doc_list).values()
    vol1 = gensim.corpora.Dictionary().load(dic_file1).values()
    vol2 = gensim.corpora.Dictionary().load(dic_file2).values()
    vol3 = gensim.corpora.Dictionary().load(dic_file3).values()
    print(len(vol1), len(vol2), len(vol3), len(map_vol))
    print("% from wiki:", len(set(vol1).intersection(map_vol))/float(len(vol1)) )
    print("% from ap:", len(set(vol2).intersection(map_vol))/float(len(vol2)) )
    print("% from merged:", len(set(vol2).intersection(map_vol))/float(len(vol3)) )

    map(model_file1, dic_file1, dist)
    map(model_file2, dic_file2, dist)
    map(model_file3, dic_file3, dist)


if __name__ == '__main__':
    main()
