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
import random

from stop_words import get_stop_words
stoplist = get_stop_words('norwegian')

dic_file1 = "lda.no.dictionary"
model_file1 = 'lda.no.model.pretrained.'
train_file1 = '../data/nowiki-latest-pages-articles.xml.bz2'
test_file1 = 'test_doc_list_wiki.json'

dic_file2 = "lda.ap.dictionary"
model_file2 = 'lda.ap.model.pretrained.'
dump_dir2 = '../data/json/10years/'
test_file2 = 'test_doc_list_aftenposten.json'

model_file3 = 'lda.merged.model.pretrained.'
dic_file3 = "lda.merged.dictionary"

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
            for line in f:
                doc = json.loads(line)
                doc_id = doc["id"]
                doc_tag = doc["tags"]
                tokens = []
                for text in doc["content"]:  # for each line in each document
                    tokens = tokens + tokenize(d, text)
                yield doc_id, doc_tag, tokens


def train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # build dictionary: preprocesing on wiki
    # remove articles that are too short(<20 words) and metadata
    doc_list = list(tokens for _, tokens in iter_wiki(train_file1))
    num_doc = len(doc_list)
    print(num_doc)
    num_train = int(num_doc * 0.8)
    random.shuffle(doc_list)
    train_doc_list1 = doc_list[: num_train]
    with open(test_file1, 'w') as outfile:
        json.dump({"test_doc_list" : doc_list[num_train:]}, outfile)
    id2word_wiki = gensim.corpora.Dictionary(train_doc_list1)
    # remove words occurs in <20 articles and >10% articles, keep 100,000 most frequent words
    id2word_wiki.filter_extremes(no_below=20, no_above=0.1)  # keep_n=100000, keep only the first keep_n most frequent tokens (or keep all if None).
    print (id2word_wiki)
    id2word_wiki.save(dic_file1)

    # build dictionary: prepocessing on aftenposten
    doc_list = list(tokens for _, _, tokens in iter_ap(dump_dir2))
    num_doc = len(doc_list)
    print(num_doc)
    num_train = int(num_doc * 0.8)
    random.shuffle(doc_list)
    train_doc_list2 = doc_list[: num_train]
    with open(test_file2, 'w') as outfile:
        json.dump({"test_doc_list" : doc_list[num_train:]}, outfile)
    id2word_ap = gensim.corpora.Dictionary(train_doc_list2)
    id2word_ap.filter_extremes(no_below=20, no_above=0.1)
    # id2word_ap = gensim.corpora.Dictionary().load(dic_file2)
    print(id2word_ap)
    id2word_ap.save(dic_file2)

    # from dictionary to corpus
    wiki_corpus = [id2word_wiki.doc2bow(tokens) for tokens in train_doc_list1]
    gensim.corpora.MmCorpus.serialize('../data/wiki_bow.mm', wiki_corpus)
    mm_corpus1 = gensim.corpora.MmCorpus('../data/wiki_bow.mm')
    print(mm_corpus1)
    ap_corpus = [id2word_ap.doc2bow(tokens) for tokens in train_doc_list2]
    gensim.corpora.MmCorpus.serialize('../data/ap_bow.mm', ap_corpus)
    mm_corpus2 = gensim.corpora.MmCorpus('../data/ap_bow.mm')
    print(mm_corpus2)

    # train the model
    train_para(mm_corpus1, id2word_wiki, model_file1)
    train_para(mm_corpus2, id2word_ap, model_file2)

    # merged corpus
    dict2_to_dict1 = id2word_wiki.merge_with(id2word_ap)
    merged_corpus = itertools.chain(mm_corpus1, dict2_to_dict1[mm_corpus2])
    id2word_wiki.save(dic_file3)
    print(id2word_wiki)
    gensim.corpora.MmCorpus.serialize('../data/merged_bow.mm', merged_corpus)
    mm_corpus3 = gensim.corpora.MmCorpus('../data/merged_bow.mm')
    print(mm_corpus3)

    train_para(mm_corpus3, id2word_wiki, model_file3)

def train_para(mm_corpus, dictionary, model_file):
    Knum = np.arange(10, 500, 40) # number of topics
    for K in Knum:
        print ("Train: " + str(model_file) + str(K))
        lda = gensim.models.ldamodel.LdaModel(corpus=mm_corpus, id2word=dictionary, num_topics=K, alpha='auto', eta='auto')
        lda.save(model_file + str(K))

if __name__ == '__main__':
    train()
