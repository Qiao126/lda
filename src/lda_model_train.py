import logging
import gensim
import bz2
import pyLDAvis.gensim
from gensim import corpora
import itertools
from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
import enchant
from scipy import stats
import numpy as np
import os
import json

from stop_words import get_stop_words

stoplist = get_stop_words('norwegian')
# dic_file = "lda.no.dictionary"
# model_file = 'lda.no.model.pretrained.'
# train_file = '../data/nowiki-latest-pages-articles.xml.bz2'

dic_file = "lda.ap.dictionary"
model_file = 'lda.ap.model.pretrained.' 
train_dir = '../data/json/train/'


def tokenize(d, text):
    a = []
    try:
        for token in simple_preprocess(text):
            if not d.check(token):
                if token not in stoplist:
                    a.append(token)
    except Exception as e: 
        print(e)
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


def iter_ap(dump_dir):  # train on several files, different when evaluation
    """Yield each article from the VG dump, as a `(doc-id, tokens)` 2-tuple."""
    d = enchant.Dict("en_US")
    for dump_file in os.listdir(dump_dir):
        path = dump_dir + dump_file
        with open(path) as f:
            i = 0
            for line in f:
                i += 1
                doc = json.loads(line)
                docid = doc["id"]
                tokens = []
                for text in doc["content"]:  # for each line in each document
                    tokens = tokens + tokenize(d, text)
                # if len(tokens) < 200:
                #    i -= 1
                #    continue  # ignore short articles
                yield docid, tokens
            # print ("num_docs: " + str(i))


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


class ApCorpus(object):
    def __init__(self, dump_dir, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` aftenposten documents from dir `dump_dir`.
        Yield each document in turn, as a list of tokens (unicode strings).
        """
        self.dump_dir = dump_dir
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        self.docs = []
        for docid, tokens in itertools.islice(iter_ap(self.dump_dir), self.clip_docs):
            self.docs.append(docid)
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs


def train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # build dictionary: preprocesing on wiki
    """
    # remove articles that are too short(<20 words) and metadata
    doc_stream = (tokens for _, tokens in iter_wiki(train_docs))
    id2word_wiki = gensim.corpora.Dictionary(doc_stream)

    # remove words occurs in <20 articles and >10% articles, keep 100,000 most frequent words
    id2word_wiki.filter_extremes(no_below=20, no_above=0.1)  # keep_n=100000, keep only the first keep_n most frequent tokens (or keep all if None).

    print (id2word_wiki)
    id2word_wiki.save(dic_file)
    """

    # build dictionary: prepocessing on aftenposten
    #doc_stream = (tokens for _, tokens in iter_ap(train_dir))
    #id2word_ap = gensim.corpora.Dictionary(doc_stream)
    id2word_ap = gensim.corpora.Dictionary().load(dic_file)
    id2word_ap.filter_extremes(no_below=20, no_above=0.1) 
    print id2word_ap
    id2word_ap.save(dic_file)

    # from dictionary to corpus
    """"
    # wiki_corpus = WikiCorpus(train_file, id2word_wiki)
    # gensim.corpora.MmCorpus.serialize('../data/wiki_bow.mm', wiki_corpus)
    mm_corpus = gensim.corpora.MmCorpus('../data/wiki_bow.mm')
    print(mm_corpus)
    # clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)
    """
    ap_corpus = ApCorpus(train_dir, id2word_ap)
    gensim.corpora.MmCorpus.serialize('../data/ap_bow.mm', ap_corpus)
    mm_corpus = gensim.corpora.MmCorpus('../data/ap_bow.mm')
    print(mm_corpus)

    # train the model
    Knum = [10, 50, 100, 500] # number of topics
    # Knum = [10]
    for K in Knum:
        print (K)
        lda = gensim.models.ldamodel.LdaModel(corpus=mm_corpus, id2word=id2word_ap, num_topics=K, alpha='auto', eta='auto')
        lda.save(model_file + str(K))

if __name__ == '__main__':
    train()
