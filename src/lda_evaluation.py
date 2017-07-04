import logging, gensim, bz2
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
stoplist = get_stop_words('norwegian')

from lda_model_train import  model_file
from lda_model_train import  dic_file
from lda_model_train import  train_file
test_file1 = '../data/nowiki-20170501-pages-articles.xml.bz2'
test_file2 = '../data/json/aftenposten.2016.json'



def tokenize(d, text):
    a = []
    for token in simple_preprocess(text):
	if not d.check(token):
	    if token not in stoplist:
	        a.append(token)
    return a

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    d = enchant.Dict("en_US")
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(d, text)
        if len(tokens) < 200 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens
        
def iter_ap(test_file):
    """Yield each article from the VG dump, as a `(doc-id, tokens)` 2-tuple."""
    d = enchant.Dict("en_US")
    with open(test_file) as f:
        i = 0
        for line in f:
            i += 1
            doc = json.loads(line)
            docid = doc["id"]
            tokens = []
            for text in doc["content"]:
                tokens = tokens + tokenize(d, text)
            #if len(tokens) < 200:
            #    i -= 1
            #    continue  # ignore short articles 
            yield docid, tokens
        #print ("num_docs: " + str(i))
        
def intra_inter(dictionary, model, test_docs, num_pairs=10000):
    # split each test document into two halves and compute topics for each half
    part1 = [model[dictionary.doc2bow(tokens[: len(tokens) / 2])] for tokens in test_docs]
    part2 = [model[dictionary.doc2bow(tokens[len(tokens) / 2 :])] for tokens in test_docs]
    
    # print computed similarities (uses cossim)
    print("average cosine similarity between corresponding parts (higher is better):")
    rel = np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)])
    print(rel)

    random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
    print("average cosine similarity between 10,000 random parts (lower is better):")    
    irel = np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs])
    print(irel)
    
    score = rel*0.5 + (1-irel)*0.5
    print("score: ", score)
    
def eval_size(dictionary, model, K):
    size = []
    num_tokens = len(dictionary.keys())
    for i in range(K):
        tuples = model.get_topic_terms(i, topn = num_tokens)
        prob = np.asarray([j[1] for j in tuples])
        prob = prob[prob > (1.0/num_tokens)]  
        size.append(len(prob))   # number of tokens for each topic
    
    size = np.asarray(size)
    mask = [int(x > float(num_tokens)/K) for x in size]  #good topics, the bigger size is, the better
    score = np.mean(mask)  # percentage of good topics
    print("score: ", score)
    
def corpus_similarity(dictionary, model, K):
    #divergence = []
    sim = []
    copy = []
    num_tokens = len(dictionary.keys())
    corpus_distribution = dictionary.doc2bow(dictionary.values()) # return a list of (word_id, word_frequency) 2-tuples.
    freqs = np.asarray([x[1] for x in corpus_distribution])
    sum_tokens = np.sum(freqs)
    for x in corpus_distribution:
        tmp = (x[0], x[1]/float(sum_tokens))  # a list of (word_id, word_prob) 2-tuples.
        copy.append(tmp)
    corpus_distribution = copy
    
    for i in range(K):
        probs = [None] * num_tokens
        topic_distribution = model.get_topic_terms(i, topn = num_tokens)  # return a list of 2-tuples (word-id, prob)
        
        # calculate the similarity between each topic and the whole corpus
        cossim = gensim.matutils.cossim(corpus_distribution, topic_distribution)
        sim.append(cossim)
    
    score = 1 - np.mean(sim)   # smaller similarity, the better
    print("score: ", score)
    
def within_doc_rank(dictionary, model, K, test_docs):
    scores = []
    top_hash = {}
    topic_doc = {}
    
    for tokens in test_docs:   
	topics = model.get_document_topics(dictionary.doc2bow(tokens)) #list of (topic-id, prob)
	num_topics = len(topics)
	for tid, _ in topics: #topic-id
		if not tid in topic_doc:
			topic_doc[tid] = 1
		else:
			topic_doc[tid] += 1
	top_topic = topics[0][0] 
	if not top_topic in top_hash:
		top_hash[top_topic] = 1
	else:
		top_hash[top_topic] += 1

    for t in topic_doc.keys():    
        if not t in top_hash:
		top_hash[t] = 0
	representitive = float(top_hash[t] + 1)/(topic_doc[t] + K)     #smoothing
        #print (t, top_hash[t], topic_doc[t], representitive)
	scores.append(representitive)
    mask =  [int(x > 1.0/K) for x in scores]     # percentage of good topics, the more predominant topic in less docs, the better
    score = np.mean(mask)
    print("score: ", score)

def coherence(dictionary, model, K, valid_docs):
    #build up the term-doc matrix
    td_hash = collections.defaultdict(dict)
    doc_max = {}
    docid = 0
    for doc in valid_docs:
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
    #transform to tf-idf term-doc matrix
    for wordid in td_hash.keys():
        num_doc = len(td_hash[wordid].keys())
        for docid in td_hash[wordid].keys():
            td_hash[wordid][docid] = (td_hash[wordid][docid] / float(doc_max[docid]) +0.5) * math.log(tot_doc/float(num_doc))
    
    coherences = []
    for i in range(K):
        top_words = [x[0] for x in model.get_topic_terms(i)]
        coherence = 0
        for j in range(len(top_words)):
            doc_j = td_hash[top_words[j]].keys()
            sum2 = 0.0
            for doc in doc_j:
                sum2 += td_hash[top_words[j]][doc]
            sum2 += 1 #smoothing, when the top word of current topic is not observed in test_docs
            for k in range(j+1, len(top_words)):
                doc_k = td_hash[top_words[k]].keys()
                co_doc = list(set(doc_j) & set(doc_k))
                sum1 = 0.0
                for doc in co_doc:
                    sum1 += td_hash[top_words[j]][doc] * td_hash[top_words[k]][doc]
                sum1 += 1 #smoothing
                coherence += math.log(sum1/sum2) 
        #print(str(i) + " cohenrence: " + str(coherence))
        coherences.append(coherence)
    score = np.mean(coherences)      
    print("score: ", score)
     
     
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

id2word_wiki = gensim.corpora.Dictionary().load(dic_file)
print(id2word_wiki)

# from dictionary to corpus

#wiki_corpus = WikiCorpus('../data/nowiki-latest-pages-articles.xml.bz2', id2word_wiki)
#gensim.corpora.MmCorpus.serialize('../data/wiki_bow.mm', wiki_corpus)
mm_corpus = gensim.corpora.MmCorpus('../data/wiki_bow.mm')
print(mm_corpus)

#clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)           
                                                                                                                                                                                  
Knum = [10, 50, 100, 500] # number of topics

for K in Knum: 
    lda = gensim.models.ldamodel.LdaModel.load(model_file + str(K))
    lda.print_topics(-1)
    
    # evaluate on 1k wiki documents **not** used in LDA training(wiki)
    doc_stream1 = (tokens for _, tokens in iter_wiki(test_file1))  # generator
    test_docs1 = list(itertools.islice(doc_stream1, 8000, 9000)) #test on 1k unkown docs
        
    # evaluate on 1k aftenposten documents **not** used in LDA training(wiki)
    doc_stream2 = (tokens for _, tokens in iter_ap(test_file2))  # generator
    test_docs2 = list(itertools.islice(doc_stream2, 8000, 9000))#test on 1k unkown docs
    
    # doc_stream3 =  (tokens for _, tokens in iter_wiki(train_file))
    # valid_docs3 = list(doc_stream3) #validate on known docs

    print("train-wiki, test-wiki: Based on LDA cosine-similarity results:")
    intra_inter(id2word_wiki, lda, test_docs1)
    
    print ("train-wiki, test-wiki: Based on LDA within-doc-rank results:")
    within_doc_rank(id2word_wiki, lda, K, test_docs1)
    """
    print ("train-wiki, test-wiki: Based on LDA semantic coherence results:")
    coherence(id2word_wiki, lda, K, test_docs1)
    
	print ("train-wiki: Based on LDA topic-size results:")
	eval_size(id2word_wiki, lda, K)

	print ("train-wiki: Based on LDA corpus-similarity results:")
	corpus_similarity(id2word_wiki, lda, K)
    """  	
    print("train-wiki, test-af: Based on LDA cosine-similarity results:")
    intra_inter(id2word_wiki, lda, test_docs2)
    
    print ("train-wiki, test-af: Based on LDA within-doc-rank results:")
    within_doc_rank(id2word_wiki, lda, K, test_docs2)

    #print ("train-wiki, test-af: Based on LDA semantic coherence results:")
    #coherence(id2word_wiki, lda, K, test_docs2)
