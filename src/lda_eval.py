import logging, gensim, bz2
import pyLDAvis.gensim
from gensim import corpora

import itertools
from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
import enchant
from scipy import stats
import numpy as np
import math

from stop_words import get_stop_words

stoplist = get_stop_words('norwegian')


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

class WikiCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).
        
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs

def intra_inter(model, test_docs, num_pairs=10000):
    # split each test document into two halves and compute topics for each half
    part1 = [model[id2word_wiki.doc2bow(tokens[: len(tokens) / 2])] for tokens in test_docs]
    part2 = [model[id2word_wiki.doc2bow(tokens[len(tokens) / 2 :])] for tokens in test_docs]
    
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
    
def eval_size(model, K):
    size = []
    num_tokens = len(id2word_wiki.keys())
    for i in range(K):
        tuples = model.get_topic_terms(i, topn = num_tokens)
        prob = np.asarray([j[1] for j in tuples])
        prob = prob[prob > (1.0/num_tokens)]
        size.append(len(prob))   #number of tokens for each topic
    size = np.asarray(size)
    mask = [int(x > float(num_tokens)/K) for x in size]  #good topics
    #score = 1 - np.mean(size / float(num_tokens))
    score = np.mean(mask)  #proportion of good topics
    print("score: ", score)
    
def tuple2list(tl):
    length = len(tl)
    res = np.zeros(length)
    for i in range(length):
        wordid, value = tl[i]
        np.put(res, wordid, value)
    return res
    
def corpus_similarity(model):
    divergence = []
    num_tokens = len(id2word_wiki.keys())
    corpus_distribution = tuple2list(id2word_wiki.doc2bow(id2word_wiki.values()))
    corpus_distribution = corpus_distribution / float(np.sum(corpus_distribution))
    for i in range(K):
        probs = [None] * num_tokens
        topic_distribution = model.get_topic_terms(i, topn = num_tokens)
        topic_distribution = tuple2list(topic_distribution)
        
        # calculate the similarity between each topic and the whole corpus
        kl = stats.entropy(corpus_distribution, topic_distribution) #distribution difference based on kl divergence
        divergence.append(kl)
        print (kl)
        
    score = np.mean(divergence)
    print("score: ", score)
    
def within_doc_rank(model, K, test_docs):
    scores = []
    top_hash = {}
    topic_doc = {}
    
    for tokens in test_docs:   
	topics = model.get_document_topics(id2word_wiki.doc2bow(tokens)) #list of (topic-id, prob)
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
    print (t, top_hash[t], topic_doc[t], representitive)
	scores.append(representitive)

    # proportion of good topics
    score = np.mean(scores)
    print("score: ", score)

def coherence(dictionary, model, K, test_docs):
    #build up the term-doc matrix
    td_hash = {}
    doc_max = {}
    docid = 0
    for doc in test_docs:
        bow = dictionary.doc2bow(doc)
        td_hash2 = {}
        max_freq = 0
        for x in bow:
            # term-doc frequency
            td_hash2[docid] = x[1]
            td_hash[x[0]] = td_hash2
            
            # record max word frequency in this doc
            if x[1] > max_freq:
                max_freq = x[1] 
        doc_max[docid] = max_freq        
        docid += 1
        
    tot_doc = docid
    #transform to tf-idf term-doc matrix
    for wordid in td_hash.keys():
        num_doc = len(td_hash.keys())
        for docid in td_hash[wordid].keys():
            td_hash[wordid][docid] = (td_hash[wordid][docid] / float(doc_max[docid]) +0.5) * math.log(tot_doc/float(num_doc))
    
    coherences = []
    for i in range(K):
        top_words = [x[0] for x in model.get_topic_terms(i)]
        sum2 = 0
        for j in range(len(top_words)):
            doc_j = td_hash[top_words[j]].keys()
            for doc in doc_j:
                sum2 += tf_hash[top_words[j]][doc]
            sum1 = 0
            for k in range(j+1, len(top_words)):
                doc_k = td_hash[top_words[k]].keys()
                co_doc = list(set(doc_j) & set(doc_k))
                for doc in co_doc:
                    sum1 += td_hash[top_words[j]][doc] * td_hash[top_words[k]][doc] +1
        coherence += math.log(sum1/float(sum2)) 
        print(K + " cohenrence: " + cohenrence)
        coherences.append(coherence)
        
     score = np.mean(coherences)      
     print("score: ", score)
      
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# remove articles that are too short(<20 words) and metadata
doc_stream = (tokens for _, tokens in iter_wiki('../data/nowiki-latest-pages-articles.xml.bz2'))
id2word_wiki = gensim.corpora.Dictionary(doc_stream)

# remove words occurs in <20 articles and >10% articles, keep 100,000 most frequent words
id2word_wiki.filter_extremes(no_below=20, no_above=0.1) #  keep_n=100000, keep only the first keep_n most frequent tokens (or keep all if None).
print (id2word_wiki)

# from dictionary to corpus

#wiki_corpus = WikiCorpus('../data/nowiki-latest-pages-articles.xml.bz2', id2word_wiki)
#gensim.corpora.MmCorpus.serialize('../data/wiki_bow.mm', wiki_corpus)
mm_corpus = gensim.corpora.MmCorpus('../data/wiki_bow.mm')
print(mm_corpus)

# train the model
model_file = 'lda.default.no.model'
model_file = 'lda.auto.filter_3.no.model'   
#clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)           
                                                                                                                                                                                  
Knum = [10, 50, 100, 500] # number of topics
for K in Knum: 
        print (K)
	lda = gensim.models.ldamodel.LdaModel(corpus=mm_corpus, id2word=id2word_wiki, num_topics=K, alpha='auto', eta='auto' )
	lda.save(model_file)

	lda = gensim.models.ldamodel.LdaModel.load(model_file)
	lda.print_topics(-1)

	#vis_data = pyLDAvis.gensim.prepare(lda, mm_corpus, id2word_wiki)
	#pyLDAvis.save_html(vis_data, model_file + ".html")

	#pyLDAvis.show(vis_data)

	# evaluate on 1k documents **not** used in LDA training
        doc_stream = (tokens for _, tokens in iter_wiki('../data/nowiki-20170501-pages-articles.xml.bz2'))  # generator
        test_docs = list(itertools.islice(doc_stream, 8000, 9000)) #test on 1k unkown docs
	''' 
	print("Based on LDA cosine-similarity results:")
	intra_inter(lda, test_docs)
	print ("Based on LDA topic-size results:")
	eval_size(lda, K)
	print ("Based on LDA corpus-similarity results:")
	corpus_similarity(lda)
	print ("Based on LDA within-doc-rank results:")
	within_doc_rank(lda, K, test_docs)
    '''
	print ("Based on LDA semantic coherence results:")
	within_doc_rank(id2word_wiki, lda, K, test_docs)

