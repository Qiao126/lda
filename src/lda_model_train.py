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
import re

from stop_words import get_stop_words
stoplist = get_stop_words('norwegian')

mcorpus_file3 = '../data/merged_bow.mm'
#mcorpus_file2 = '../data/ap_bow.mm'
mcorpus_file2 = '../data/ap2_bow.mm'
mcorpus_file1 = '../data/wiki_bow.mm'

dic_file1 = "lda.no.dictionary"
model_file1 = 'lda.no.model.pretrained.'
train_file1 = '../data/nowiki-latest-pages-articles.xml.bz2'
train_wiki_file = 'train_doc_list_wiki.json'
test_file1 = 'test_doc_list_wiki.json'

#dic_file2 = "lda.ap.dictionary"
#model_file2 = 'lda.ap.model.pretrained.'
dic_file2 = "lda.ap2.dictionary"
model_file2 = 'lda.ap2.model.pretrained.'
#dump_dir2 = '../data/json/10years/'
dump_dir2 = '../data/output/'
train_ap_file = 'train_doc_list_aftenposten.json'
test_file2 = 'test_doc_list_aftenposten.json'

model_file3 = 'lda.merged.model.pretrained.'
dic_file3 = "lda.merged.dictionary"

test_file3 = '../data/test_doc_list_apSection.json'

SEED = 126

#with open(test_file3, 'r') as data_file:
#    testset  = json.load(data_file)['test_doc_id']


def tokenize(d, text, vol=None):
    a = []
    try:
        for token in simple_preprocess(text):
            if not d.check(token):
                if token not in stoplist:
                    if vol == None or token in vol:
                        a.append(token)
    except Exception as e:
        print(e)
    return a


def iter_wiki(dump_file, vol):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia MediaWiki Mal Hjelp Kategori Portal Fil Bruker Bok'.split()
    d = enchant.Dict("en_US")
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        #m = re.search('^(.+?):', title)
        #if m:
        #    meta = m.group(1)
        #    print (meta)
        text = filter_wiki(text)
        tokens = tokenize(d, text, vol)
        if len(tokens) < 200 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens


def iter_ap(dump_dir, mode, vol):  # train on several files, different when evaluation
    """Yield each article from the VG dump, as a `(doc-id, tokens)` 2-tuple."""
    d = enchant.Dict("en_US")
    for dump_file in os.listdir(dump_dir):
        path = dump_dir + dump_file
        #print dump_file
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc["id"]

                if mode == "train" and doc_id in testset: #skip test set
                    continue
                else:
                    doc_tag = doc["section"] #doc["tags"]
                    tokens = []
                    for text in doc["content"]:  # for each line in each document
                        tokens = tokens + tokenize(d, text, vol)
                    if len(tokens) < 100:
                        continue
                    yield doc_id, doc_tag, tokens


def train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # build dictionary: preprocesing on wiki
    # remove articles that are too short(<20 words) and metadata
    #with open("vocalbulary.txt", 'r') as data_file:
    #    vol  = json.load(data_file)['vocalbulary']
    vol = None
    """
    doc_list = list(tokens for _, tokens in iter_wiki(train_file1, vol))
    num_doc = len(doc_list)
    print(num_doc)
    num_train = int(num_doc * 0.8)
    random.seed(SEED)
    random.shuffle(doc_list)
    train_doc_list1 = doc_list[: num_train]
    with open(train_wiki_file, 'w') as outfile:
        json.dump({"train_doc_list" : train_doc_list1}, outfile)
    id2word_wiki = gensim.corpora.Dictionary(train_doc_list1)
    with open(test_file1, 'w') as outfile:
        json.dump({"test_doc_list" : doc_list[num_train:]}, outfile)
    id2word_wiki = gensim.corpora.Dictionary(train_doc_list1)
    # remove words occurs in <20 articles and >10% articles, keep 100,000 most frequent words
    id2word_wiki.filter_extremes(no_below=20, no_above=0.1)  # keep_n=100000, keep only the first keep_n most frequent tokens (or keep all if None).
    print (id2word_wiki)
    id2word_wiki.save(dic_file1)

    with open("wiki-vocalbulary.txt", 'w') as outfile:
        json.dump({"vocalbulary" : id2word_wiki.values()}, outfile)

    id2word_wiki = gensim.corpora.Dictionary().load(dic_file1)
    with open(train_wiki_file, 'r') as data_file:
        train_doc_list1  = json.load(data_file)['train_doc_list']


    with open("wiki-vocalbulary.txt", 'r') as data_file:
        vol  = json.load(data_file)['vocalbulary']
    """
    # build dictionary: prepocessing on aftenposten
    """
    doc_list = list(tokens for _, _, tokens in iter_ap(dump_dir2, "train", vol))
    num_doc = len(doc_list)
    print(num_doc)
    num_train = int(num_doc * 0.8)
    random.seed(SEED)
    random.shuffle(doc_list)
    train_doc_list2 = doc_list[: num_train]
    with open(train_ap_file, 'w') as outfile:
        json.dump({"train_doc_list" : train_doc_list2}, outfile)
    with open(test_file2, 'w') as outfile:
        json.dump({"test_doc_list" : doc_list[num_train:]}, outfile)
    """
    with open(test_file3, 'r') as data_file:
        data  = json.load(data_file)
        test_size = data['test_size']
        train_doc_list2 = data['test_doc_list'][test_size:]
    """
    random.seed(SEED)
    random.shuffle(train_doc_list2)
    fold = int(len(train_doc_list2)/20)
    train_docs = []
    for i in range(20):
        train_docs.append(train_doc_list2[:fold])
        train_doc_list2 = train_doc_list2[fold:]
    """
    """
    for i in range(len(train_docs)):
        print("fold ", i)
        id2word_ap = gensim.corpora.Dictionary(train_docs[i])
        id2word_ap.filter_extremes(no_below=20, no_above=0.1)
        # id2word_ap = gensim.corpora.Dictionary().load(dic_file2)
        print(id2word_ap)
        #id2word_ap.save(dic_file2)
        id2word_ap.save('lda.ap2.' + str(i) + '.dictionary')
        """
        """
        # from dictionary to corpus
        wiki_corpus = [id2word_wiki.doc2bow(tokens) for tokens in train_doc_list1]
        gensim.corpora.MmCorpus.serialize(mcorpus_file1, wiki_corpus)
        mm_corpus1 = gensim.corpora.MmCorpus(mcorpus_file1)
        print(mm_corpus1)
        """
        """
        mm_corpus2 = '../data/ap2_bow.' + str(i) +'.mm'
        ap_corpus = [id2word_ap.doc2bow(tokens) for tokens in train_docs[i]]
        gensim.corpora.MmCorpus.serialize(mm_corpus2, ap_corpus)
        mm_corpus2 = gensim.corpora.MmCorpus(mm_corpus2)
        print(mm_corpus2)
    """
    id2word_ap = gensim.corpora.Dictionary().load('lda.ap2' + '.dictionary')
    print(id2word_ap)
    mm_corpus2 = '../data/ap2_bow' + '.mm'
    mm_corpus2 = gensim.corpora.MmCorpus(mm_corpus2)
    print(mm_corpus2)
    # train the model
    #train_para(mm_corpus1, id2word_wiki, model_file1)
    train_para(mm_corpus2, id2word_ap, model_file2)
    """
    # merged corpus
    dict2_to_dict1 = id2word_wiki.merge_with(id2word_ap)
    merged_corpus = itertools.chain(mm_corpus1, dict2_to_dict1[mm_corpus2])
    id2word_wiki.save(dic_file3)
    print(id2word_wiki)
    gensim.corpora.MmCorpus.serialize(mcorpus_file3, merged_corpus)
    mm_corpus3 = gensim.corpora.MmCorpus(mcorpus_file3)
    print(mm_corpus3)

    train_para(mm_corpus3, id2word_wiki, model_file3)
    """
def train_para(mm_corpus, dictionary, model_file):
    #Knum = np.arange(10, 160, 10) # number of topics
    Knum = [18]
    i = 61
    for K in Knum:
        for a in [None, 'asymmetric', 'auto']:
            for b in [None, 'auto']:
                print ("Train: " + str(model_file) + str(K))
                lda = gensim.models.ldamodel.LdaModel(corpus=mm_corpus, id2word=dictionary, num_topics=K, alpha=a, eta=b)
                lda.save(model_file + str(K) + '.' + str(i))
                i += 1


if __name__ == '__main__':
    train()
