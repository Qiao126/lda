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
test_file4 = '../data/test_doc_list_apSection2.json'
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
    vol = None

    # build dictionary: prepocessing on aftenposten

    """
    with open(test_file3, 'r') as data_file:
        data  = json.load(data_file)
        test_size = data['test_size']
        test_doc_tag = data['test_doc_tag'][:test_size]
        test_doc_section = data['test_doc_tag']
        test_doc_list = data['test_doc_list']
        test_doc_id = data['test_doc_id']

    sections = OrderedDict(sorted(Counter(test_doc_tag).items()))
    dids =[]
    docs = []
    secs = []
    i = 0
    for did, tokens, section in zip(test_doc_id, test_doc_list, test_doc_section):
        if section != sections[3][0] and section != sections[15][0]:
            dids.append(did)
            docs.append(tokens)
            secs.append(section)
            i += 1

    test_size = int(0.1 * i)

    with open(test_file4, 'w') as outfile:
        json.dump({"test_size": test_size,
                    "test_doc_tag" : secs,
                    "test_doc_list": docs,
                    "test_doc_id" : dids}, outfile)
    """
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

    train_docs = test_doc_list[test_size:]
    id2word_ap = gensim.corpora.Dictionary(train_docs)
    id2word_ap.filter_extremes(no_below=20, no_above=0.1)
    id2word_ap.save('lda.ap2.dictionary')

    mm_corpus2 = '../data/ap2_bow.mm'
    ap_corpus = [id2word_ap.doc2bow(tokens) for tokens in train_docs]
    gensim.corpora.MmCorpus.serialize(mm_corpus2, ap_corpus)

    id2word_ap = gensim.corpora.Dictionary().load('lda.ap2.dictionary')
    print(id2word_ap)
    mm_corpus2 = gensim.corpora.MmCorpus('../data/ap2_bow.mm')
    print(mm_corpus2)
    # train the model
    train_para(mm_corpus2, id2word_ap, model_file2)
    
def train_para(mm_corpus, dictionary, model_file):
    #Knum = np.arange(10, 160, 10) # number of topics
    Knum = [18]
    i = 79
    for K in Knum:
        #for a in [None, 'asymmetric', 'auto']:
        for a in [100.0/K, 50.0/K, 20.0/K, 5.0/K, 1.0/K]:
            for b in [0.1, 0.002]:
                print ("Train: " + str(model_file) + str(K))
                lda = gensim.models.ldamodel.LdaModel(corpus=mm_corpus, id2word=dictionary, num_topics=K, alpha=a, eta=b)
                lda.save(model_file + str(K) + '.' + str(i))
                i += 1


if __name__ == '__main__':
    train()
