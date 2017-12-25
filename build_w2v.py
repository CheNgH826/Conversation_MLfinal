from gensim.models import Word2Vec
import jieba

corpus = []
for i in range(1,6):
    corpus.append('data/training_data/'+str(i)+'_train.txt')

sen_list = []
for one_corpus in corpus:
    with open(one_corpus, 'r') as f:
        sens = list(f)
        for sen in sens:
            sen = sen.replace('\n', '')
            sen_word_list = list(jieba.cut(sen))
            sen_word_list.append('<BOS>')
            sen_word_list.append('<EOS>')
            sen_list.append(list(sen_word_list))

WORDVEC_DIM = 300
w2v_model = Word2Vec(sen_list, min_count=1, size=WORDVEC_DIM, iter=20) 
w2v_model.save('data/w2v_model')
