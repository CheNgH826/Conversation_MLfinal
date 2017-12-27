from gensim.models import Word2Vec
import jieba
import sys
jieba.dt.cache_file = 'jieva.cache.new'

corpus = []
for i in range(1,6):
    corpus.append('training_data/'+str(i)+'_train.txt')

sen_list = []
for one_corpus in corpus:
    with open(one_corpus, 'r') as f:
        sens = list(f)
        for sen in sens:
            sen = sen.replace('\n', '')
            sen_word_list = list(jieba.cut(sen))
            #sen_word_list.append('<BOS>')
            #sen_word_list.append('<EOS>')
            sen_list.append(list(sen_word_list))

WORDVEC_DIM = int(sys.argv[1])
w2v_model = Word2Vec(sen_list, min_count=10, size=WORDVEC_DIM, iter=10) 
w2v_model.save('w2v_model/w2v_model_{}_mincount10'.format(WORDVEC_DIM))
