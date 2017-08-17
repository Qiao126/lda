import logging, gensim
import json

from gensim.utils import simple_preprocess
import enchant
import itertools
import numpy as np

from stop_words import get_stop_words
stoplist = get_stop_words('norwegian')

from lda_model_train import  model_file
from lda_model_train import  dic_file
from lda_model_train import  train_file
test_file = '../data/json/aftenposten.2016.json'

def tokenize(d, text):
    a = []
    for token in simple_preprocess(text):
	if not d.check(token):
	    if token not in stoplist:
	        a.append(token)
    return a

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
        print ("num_docs: " + str(i))
    
            
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

id2word_wiki = gensim.corpora.Dictionary().load(dic_file)
print(id2word_wiki)

K = 10
lda = gensim.models.ldamodel.LdaModel.load(model_file + str(K))
lda.print_topics(-1)

     
# evaluate on 1k aftenposten documents **not** used in LDA training(wiki)
doc_stream = (tokens for _, tokens in iter_ap(test_file))  # generator
test_docs = list(itertools.islice(doc_stream, 8000, 9000))#test on 1k unkown docs

print("Test on Aftenposten:")
print("Based on LDA cosine-similarity results:")
intra_inter(id2word_wiki, lda, test_docs)