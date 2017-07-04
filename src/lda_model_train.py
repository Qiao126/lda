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

from stop_words import get_stop_words

stoplist = get_stop_words('norwegian')
dic_file = "lda.no.dictionary"
model_file = 'lda.no.model.pretrained.'   
train_file = '../data/nowiki-latest-pages-articles.xml.bz2'

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


def train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # remove articles that are too short(<20 words) and metadata
    doc_stream = (tokens for _, tokens in iter_wiki(train_docs))
    id2word_wiki = gensim.corpora.Dictionary(doc_stream)

    # remove words occurs in <20 articles and >10% articles, keep 100,000 most frequent words
    id2word_wiki.filter_extremes(no_below=20, no_above=0.1) #  keep_n=100000, keep only the first keep_n most frequent tokens (or keep all if None).
    print (id2word_wiki)


    id2word_wiki.save(dic_file)

    # from dictionary to corpus

    #wiki_corpus = WikiCorpus('../data/nowiki-latest-pages-articles.xml.bz2', id2word_wiki)
    #gensim.corpora.MmCorpus.serialize('../data/wiki_bow.mm', wiki_corpus)
    mm_corpus = gensim.corpora.MmCorpus('../data/wiki_bow.mm')
    print(mm_corpus)

    # train the model
    #clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)           

    Knum = [10, 50, 100, 500] # number of topics
    for K in Knum: 
        print (K)
        lda = gensim.models.ldamodel.LdaModel(corpus=mm_corpus, id2word=id2word_wiki, num_topics=K, alpha='auto', eta='auto' )
        lda.save(model_file + str(K))

if __name__ == '__main__':
    train()

