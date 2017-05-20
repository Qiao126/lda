import logging, gensim, bz2
import pyLDAvis.gensim
from gensim import corpora

import itertools
from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
import enchant
from scipy import stats

from stop_words import get_stop_words

stoplist = get_stop_words('norwegian')

'''
stoplist = ["alle","at","av","bare","begge","ble","blei","bli","blir","blitt","da","de","deg",
"dei","deim","deira","deires","dem","den","denne","der","dere","deres","det","dette","di","din","disse","ditt","du",
"dykk","dykkar","eg","ein","eit","eitt","eller","elles","en","enn","er","et","ett","etter","for","fordi","fra",
"ha","hadde","han","hans","har","hennar","henne","hennes","her","ho","hoe","honom","hoss","hossen","hun",
"hva","hvem","hver","hvilke","hvilken","hvis","hvor","hvordan","hvorfor","i","ikke","ikkje","ingen","ingi","inkje","inn",
"inni","ja","jeg","kan","kom","korleis","korso","kun","kunne","kva","kvar","kvarhelst","kven","kvi","kvifor","man","mange",
"me","med","medan","meg","meget","mellom","men","mi","min","mine","mitt","mot","mykje","ned","no","noe","noen",
"noka","noko","nokon","nokor","nokre","og","om","opp","oss","over","samme","seg","selv","si","sia",
"sidan","siden","sin","sine","sitt","skal","skulle","slik","so","som","somme","somt","til","um","upp","ut","uten",
"var","vart","varte","ved","vere","verte","vi","vil","ville","vore","vors","vort"]
'''

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
K = 10 # number of topics
lda = gensim.models.ldamodel.LdaModel(corpus=mm_corpus, id2word=id2word_wiki, num_topics=K, alpha='auto', eta='auto' )
lda.save(model_file)

lda = gensim.models.ldamodel.LdaModel.load(model_file)
lda.print_topics(-1)

vis_data = pyLDAvis.gensim.prepare(lda, mm_corpus, id2word_wiki)

pyLDAvis.save_html(vis_data, model_file + ".html")
#pyLDAvis.show(vis_data)

# evaluate on 1k documents **not** used in LDA training
doc_stream = (tokens for _, tokens in iter_wiki('../data/nowiki-20170501-pages-articles.xml.bz2'))  # generator
test_docs = list(itertools.islice(doc_stream, 8000, 9000))

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
    
def eval_size(model):
    size = []
    num_tokens = len(id2word_wiki.keys())
    for i in range(K):
        size.append(len(get_topic_terms(i, topn = num_tokens)))
    size = np.asarray(size)
    score = 1 - np.mean(size / num_tokens)
    print("score: ", score)
    
def tuple2list(tl):
    length = len(tl)
    res = np.zeros(length)
    for i in range(length):
        wordid, value = length[i]
        np.put(res, wordid, value)
    return res
    
def corpus_similarity(model):
    distance = []
    num_tokens = len(id2word_wiki.keys())
    corpus_distribution = id2word_wiki.items()
    corpus_distribution = np.mean(tuple2list(corpus_distribution), axis=1)

    for i in range(K):
        probs = [None] * num_tokens
        topic_distribution = get_topic_terms(i, topn = num_tokens)
        topic_distribution = tuple2list(topic_distribution)
        
        # calculate the similarity between each topic and the whole corpus
        kl = stats.entropy(corpus_distribution, topic_distribution) #distribution difference based on kl distance
        distance = distance.append(kl)
        
    score = np.mean(distance)
    print("score: ", score)
    

print("Based on LDA cosine-similarity results:")
intra_inter(lda, test_docs)
print ("Based on LDA topic-size results:")
eval_size(lda)
print ("Based on LDA corpus-similarity results:")
corpus_similarity(lda)