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
import datetime

from lda_model_train import model_file1, model_file2, model_file3
from lda_model_train import dic_file1, dic_file2, dic_file3
from lda_model_train import test_file1, test_file2, test_file3
from lda_model_train import mcorpus_file3, mcorpus_file2, mcorpus_file1
from lda_model_train import train_wiki_file, train_ap_file
from lda_model_train import SEED
from lda_model_train import tokenize
stoplist = get_stop_words('norwegian')


def intra_inter(dictionary, model, test_docs):

    # split each test document into two halves and compute topics for each half
    part1 = [model[dictionary.doc2bow(tokens[: len(tokens) / 2])] for tokens in test_docs]
    part2 = [model[dictionary.doc2bow(tokens[len(tokens) / 2:])] for tokens in test_docs]

    # print computed similarities (uses cossim)
    #print("average cosine similarity between corresponding parts (higher is better):")
    rel = np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)])
    print("{:.4f}".format(rel))

    num_pairs = len(test_docs)
    random_pairs = np.random.randint(0, len(test_docs), size=(len(test_docs), 2))
    #print("average cosine similarity between 10,000 random parts (lower is better):")
    irel = np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs])
    print("{:.4f}".format(irel))

    diff = rel - irel
    print("{:.4f}".format(diff))


def eval_size(dictionary, corpus_dist, model, K):
    size = []
    tot = np.sum(corpus_dist.values())
    for i in range(K):
        count = 0
        for (wid, prob) in model.get_topic_terms(i): # topn=10
            count += corpus_dist[wid]
            #print(wid, prob, corpus_dist[wid])
        #print("--------------------")
        size.append(count)   # number of tokens for each topic
    #print(np.mean(size), tot)
    score = np.mean(size)/float(tot)
    print("{:.8f}".format(score))

def prep_corpusdiff(corpus_dist):
    corpus_distribution = []
    freqs = corpus_dist.values()
    sum_tokens = np.sum(freqs)
    for x in corpus_dist:
        tmp = (x, corpus_dist[x]/float(sum_tokens))  # a list of (word_id, word_prob) 2-tuples.
        corpus_distribution.append(tmp)
    return corpus_distribution

def corpus_difference(dictionary, corpus_distribution, model, K): # same as topic size
    sim = []
    num_tokens = len(dictionary.keys())
    #print(num_tokens, len(corpus_dist.keys()))

    for i in range(K):
        probs = [None] * num_tokens
        topic_distribution = model.get_topic_terms(i, topn=num_tokens)  # return a list of 2-tuples (word-id, prob)
        # calculate the similarity between each topic and the whole corpus
        cossim = gensim.matutils.cossim(corpus_distribution, topic_distribution)
        sim.append(cossim)

    score = 1 - np.mean(sim)   # smaller similarity, the better
    print("{:.4f}".format(score))


def within_doc_rank(dictionary, model, K, corpus_docs):
    scores = []
    top_hash = {}
    topic_doc = {}

    for tokens in corpus_docs:

        topics = model.get_document_topics(dictionary.doc2bow(tokens), minimum_probability=1.0/K)  # list of (topic-id, prob)
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
            #print(top_hash[t], topic_doc[t])
            representitive = float(top_hash[t]) / topic_doc[t]   # no smoothing

        scores.append(representitive)
    score = np.mean(scores)
    print("{:.4f}".format(score))

def prep_coherence(mm_corpus):
    # build up the term-doc matrix
    td_hash = collections.defaultdict(dict)
    doc_max = {}
    docid = 0
    for doc in mm_corpus:
        #bow = dictionary.doc2bow(doc)
        max_freq = 0
        for (wid, freq) in doc:
            # term-doc frequency
            td_hash[wid][docid] = freq

            # record max word frequency in this doc
            if freq > max_freq:
                max_freq = freq
        doc_max[docid] = max_freq
        docid += 1

    tot_doc = docid
    # transform to tf-idf term-doc matrix
    for wordid in td_hash.keys():
        num_doc = len(td_hash[wordid].keys())
        for docid in td_hash[wordid].keys():
            td_hash[wordid][docid] = (td_hash[wordid][docid] / float(doc_max[docid]) + 0.5) * math.log(tot_doc / float(num_doc))
    return td_hash, doc_max


def coherence(dictionary, model, K, td_hash, doc_max):
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


def perplexity(dictionary, model, test_docs):
    chunk = []
    for i, doc in enumerate(test_docs):
        chunk.append(dictionary.doc2bow(doc))
    print("{:.4f}".format(model.log_perplexity(chunk, total_docs=None)))


def eval(dictionary, mm_corpus, corpus_dist, corpus_docs, model_file, test_docs3, i): # fixed vocalbulary, ap-corpus, and diff models
    print(dictionary)
    print(mm_corpus)
    corpus_distribution = prep_corpusdiff(corpus_dist)
    td_hash, doc_max = prep_coherence(mm_corpus)
    #Knum = np.arange(10, 500, 40)  # number of topics
    #Knum = np.arange(10, 160, 10)
    Knum = [18]
    for K in Knum:
        print("Train on: " + str(model_file) + str(K) + "---------------------------------------", i)
        lda = gensim.models.ldamodel.LdaModel.load(model_file + str(K) + '.' + str(i))
        print("inter-intra", datetime.datetime.now())
        intra_inter(dictionary, lda, test_docs3)
        print("size", datetime.datetime.now())
        eval_size(dictionary, corpus_dist, lda, K)
        print("corpus_diff", datetime.datetime.now())
        corpus_difference(dictionary, corpus_distribution, lda, K)
        print("within_doc_rank", datetime.datetime.now())
        within_doc_rank(dictionary, lda, K, corpus_docs)
        print("coherence", datetime.datetime.now())
        coherence(dictionary, lda, K, td_hash, doc_max)
        print("perplexity", datetime.datetime.now())
        perplexity(dictionary, lda, test_docs3)

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open(test_file3, 'r') as data_file:
        data  = json.load(data_file)
        test_size = data['test_size']
        train_doc_list2 = data['test_doc_list'][test_size:]
        test_docs3 = data['test_doc_list'][:test_size]
    """
    random.seed(SEED)
    random.shuffle(train_doc_list2)
    fold = int(len(train_doc_list2)/20)
    train_docs = []
    for i in range(20):
        train_docs.append(train_doc_list2[:fold])
        train_doc_list2 = train_doc_list2[fold:]
    """
    ap_docs = train_doc_list2
    dictionary2 = gensim.corpora.Dictionary().load('lda.ap2.dictionary')

    corpus_dist2 = {}
    mm_corpus2 = gensim.corpora.MmCorpus('../data/ap2_bow.mm')

    print(mm_corpus2)

    for doc in mm_corpus2:
        for (wid, freq) in doc:
             if wid in corpus_dist2:
                corpus_dist2[wid] += freq
             else:
                corpus_dist2[wid] = freq

    print(len(corpus_dist2.keys()), np.sum(corpus_dist2.values()))

    """

    # evaluate on 1k aftenposten documents **not** used in LDA training(wiki)
    with open(test_file2, 'r') as data_file:
        test_docs_list  = json.load(data_file)['test_doc_list']
    test_docs2 = test_docs_list  # test on 1/5 aftenposten docs
    """

    for i in range(61, 67):
        eval(dictionary2, mm_corpus2, corpus_dist2, ap_docs, model_file2, test_docs3, i)  #ap->ap(train/test corpus)
