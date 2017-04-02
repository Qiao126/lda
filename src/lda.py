import logging, gensim, bz2
import pyLDAvis.gensim
import copy 
from gensim.models import VocabTransform
from gensim import corpora

def filter_corpus(old_dict, old_corpus):
    # filter the dictionary
    new_dict = copy.deepcopy(old_dict)
    new_dict.filter_n_most_frequent(int(0.005 * len(new_dict)))
    new_dict.save('filtered.dict')

    # now transform the corpus
    old2new = {old_dict.token2id[token]:new_id for new_id, token in new_dict.iteritems()}
    vt = VocabTransform(old2new)
    filtered_mm = 'filtered_corpus.mm'
    corpora.MmCorpus.serialize(filtered_mm, vt[old_corpus], id2word=new_dict)
    return new_dict, gensim.corpora.MmCorpus(filtered_mm)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File('result/wiki_no/_wordids.txt.bz2'))

# load corpus iterator
mm = gensim.corpora.MmCorpus('result/wiki_no/_tfidf.mm')

# filter the corpus
id2word, mm = filter_corpus(id2word, mm)

# train the model
#model_file = 'lda.default.no.model'
model_file = 'lda.auto.filter_0_0_105.no.model'
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, alpha='auto', chunksize=10000, passes=1)
lda.save(model_file)

lda = gensim.models.ldamodel.LdaModel.load(model_file)
lda.print_topics(-1)

vis_data = pyLDAvis.gensim.prepare(lda, mm, id2word)

pyLDAvis.save_html(vis_data, model_file + ".html")
#pyLDAvis.show(vis_data)

