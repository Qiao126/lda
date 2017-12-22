import logging
import gensim
import bz2
import pyLDAvis.gensim
from gensim import corpora
import json
import itertools
from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
import enchant
from scipy import stats
import numpy as np
import math
import collections
import random
from stop_words import get_stop_words
from operator import itemgetter

from lda_model_train import model_file1
from lda_model_train import dic_file1
from lda_model_train import model_file2
from lda_model_train import dic_file2
from lda_model_train import model_file3
from lda_model_train import dic_file3

from lda_model_train import test_file1
from lda_model_train import test_file2

from lda_model_train import tokenize
stoplist = get_stop_words('norwegian')

mcorpus_file3 = '../data/merged_bow.mm'
mcorpus_file2 = '../data/ap_bow.mm'
mcorpus_file1 = '../data/wiki_bow.mm'


def intra_inter(dictionary, model, test_docs, num_pairs=10000):
    test_docs = test_docs[:num_pairs]
    # split each test document into two halves and compute topics for each half
    part1 = [model[dictionary.doc2bow(tokens[: len(tokens) / 2])] for tokens in test_docs]
    part2 = [model[dictionary.doc2bow(tokens[len(tokens) / 2:])] for tokens in test_docs]

    # print computed similarities (uses cossim)
    #print("average cosine similarity between corresponding parts (higher is better):")
    rel = np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)])
    print("{:.4f}".format(rel))

    num_pairs = len(test_docs)
    random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
    #print("average cosine similarity between 10,000 random parts (lower is better):")
    irel = np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs])
    print("{:.4f}".format(irel))


    # instead of random parts similarity, calculate the similarity of every two topics, the lower, the less topics are overlapping.

def eval_size(dictionary, corpus_dist, model, K):
    size = []

    for i in range(K):
        count = 0
        for (wid, prob) in model.get_topic_terms(i): # topn=10
            count += corpus_dist[wid]
        size.append(count)   # number of tokens for each topic

    score = np.mean(size)
    print("{:.4f}".format(score))


def corpus_difference(dictionary, corpus_dist, model, K):
    sim = []
    copy = []
    num_tokens = len(dictionary.keys())
    #print(num_tokens, len(corpus_dist.keys()))

    freqs = corpus_dist.values()
    sum_tokens = np.sum(freqs)
    for x in corpus_dist:
        tmp = (x, corpus_dist[x]/float(sum_tokens))  # a list of (word_id, word_prob) 2-tuples.
        copy.append(tmp)
    corpus_distribution = copy

    for i in range(K):
        probs = [None] * num_tokens
        topic_distribution = model.get_topic_terms(i, topn=num_tokens)  # return a list of 2-tuples (word-id, prob)
        # calculate the similarity between each topic and the whole corpus
        cossim = gensim.matutils.cossim(corpus_distribution, topic_distribution)
        sim.append(cossim)

    score = 1 - np.mean(sim)   # smaller similarity, the better
    print("{:.4f}".format(score))


def within_doc_rank(dictionary, model, K, test_docs):
    scores = []
    top_hash = {}
    topic_doc = {}

    for tokens in test_docs:
        topics = model.get_document_topics(dictionary.doc2bow(tokens), minimum_probability=float(1/K))  # list of (topic-id, prob)
        num_topics = len(topics)
        topics = sorted(topics, key=itemgetter(1), reverse=True) # order by desc prob
        if num_topics == 0:
            continue
        for tid, _ in topics:  # topic-id
            if tid not in topic_doc:
                topic_doc[tid] = 1
            else:
                topic_doc[tid] += 1   # increase the count of doc where it is assigned to
        top_topic = topics[0][0]  #  get the largest prob
        if top_topic not in top_hash:
            top_hash[top_topic] = 1
        else:
            top_hash[top_topic] += 1  # increase the count of doc where it is the top 1

    for t in topic_doc.keys():
        if t not in top_hash:
            representitive =  0;
        else:
            representitive = float(top_hash[t] / topic_doc[t])   # no smoothing
            print (t, top_hash[t], topic_doc[t], representitive)
        scores.append(representitive)
    score = np.mean(scores)
    print("{:.4f}".format(score))


def coherence(dictionary, model, K, test_docs):
    # build up the term-doc matrix
    td_hash = collections.defaultdict(dict)
    doc_max = {}
    docid = 0
    for doc in test_docs:
        bow = dictionary.doc2bow(doc)

        max_freq = 0
        for x in bow:
            # term-doc frequency
            td_hash[x[0]][docid] = x[1]

            # record max word frequency in this doc
            if x[1] > max_freq:
                max_freq = x[1]
        doc_max[docid] = max_freq
        docid += 1

    tot_doc = docid
    # transform to tf-idf term-doc matrix
    for wordid in td_hash.keys():
        num_doc = len(td_hash[wordid].keys())
        for docid in td_hash[wordid].keys():
            td_hash[wordid][docid] = (td_hash[wordid][docid] / float(doc_max[docid]) + 0.5) * math.log(tot_doc / float(num_doc))
    coherences = []
    for i in range(K):
        top_words = [x[0] for x in model.get_topic_terms(i)]  #topn=10
        coherence = 0
        for j in range(len(top_words)):
            doc_j = td_hash[top_words[j]].keys()
            sum2 = 0.0
            for doc in doc_j:
                sum2 += td_hash[top_words[j]][doc]
            sum2 += 1  # smoothing, when the top word of current topic is not observed in test_docs
            for k in range(j+1, len(top_words)):
                doc_k = td_hash[top_words[k]].keys()
                co_doc = list(set(doc_j) & set(doc_k))
                sum1 = 0.0
                for doc in co_doc:
                    sum1 += td_hash[top_words[j]][doc] * td_hash[top_words[k]][doc]
                sum1 += 1  # smoothing
                coherence += math.log(sum1/sum2)
        # print(str(i) + " cohenrence: " + str(coherence))
        coherences.append(coherence)
    score = np.mean(coherences)
    print("{:.4f}".format(score))


def eval(dic_file, mcorpus_file, model_file):
    dictionary = gensim.corpora.Dictionary().load(dic_file)
    print(dictionary)

    corpus_dist = {}
    mm_corpus2 = gensim.corpora.MmCorpus(mcorpus_file)
    for doc in mm_corpus2:
        for (wid, freq) in doc:
            if wid in corpus_dist:
                corpus_dist[wid] += freq
            else:
                corpus_dist[wid] = freq

    """
    # evaluate on 1k wiki documents **not** used in LDA training(wiki)
    with open(test_file1, 'r') as data_file:
        test_docs_list  = json.load(data_file)['test_doc_list']
    test_docs1 = test_docs_list  # test on random 1/5 20170501-wiki docs
    """
    # evaluate on 1k aftenposten documents **not** used in LDA training(wiki)
    with open(test_file2, 'r') as data_file:
        test_docs_list  = json.load(data_file)['test_doc_list']
    test_docs2 = test_docs_list  # test on 1/5 aftenposten docs

    #Knum = np.arange(10, 500, 40)  # number of topics
    Knum = [10]
    for K in Knum:
        print("Train on: " + str(model_file) + str(K) + "---------------------------------------")
        lda = gensim.models.ldamodel.LdaModel.load(model_file + str(K))
        """
        print("Test on: wiki, cosine-similarity:")
        intra_inter(dictionary, lda, test_docs1)
        print("***************************************************************")

        print ("Test on: wiki, within-doc-rank:")
        within_doc_rank(dictionary, lda, K, test_docs1)
        print("***************************************************************")

        print ("Test on: wiki,semantic coherence")
        coherence(dictionary, lda, K, test_docs1)
        print("***************************************************************")
        """

        #print("Test on: af, cosine-similarity results:")
        #intra_inter(dictionary, lda, test_docs2)
        #print("***************************************************************")

        #print ("topic-size:")
        #eval_size(dictionary, corpus_dist, lda, K)
        #print("***************************************************************")

        #print ("corpus-difference:")
        #corpus_difference(dictionary, corpus_dist, lda, K)
        #print("***************************************************************")

        #print ("Test on: af: Based on LDA within-doc-rank results:")
        within_doc_rank(dictionary, lda, K, test_docs2)
        #print("***************************************************************")

        #print ("Test on: af: Based on LDA semantic coherence results:")
        #coherence(dictionary, lda, K, test_docs2)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    eval(dic_file1, mcorpus_file1, model_file1)  #wiki->ap
    eval(dic_file2, mcorpus_file2, model_file2)  #ap->ap
    eval(dic_file3, mcorpus_file3, model_file3)  #merged->ap
