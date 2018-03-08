from lda_model_train import model_file2
import pyLDAvis.gensim
import gensim


dictionary2 = gensim.corpora.Dictionary().load('lda.ap2.dictionary')
mm_corpus2 = gensim.corpora.MmCorpus('../data/ap2_bow.mm')

for i in range(61, 67):
    K = 18
    lda = gensim.models.ldamodel.LdaModel.load(model_file2 + str(K) + '.' + str(i))
    vis_data = pyLDAvis.gensim.prepare(lda, mm_corpus2, dictionary2)
    pyLDAvis.save_html(vis_data, model_file2 + ".html")
