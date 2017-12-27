import jieba
jieba.dt.cache_file = 'jieva.cache.new'
import numpy as np
from util import DataManager,Vocabulary

max_word_len=14
word_dim_list = [50, 100, 150, 200, 250, 300, 350, 400]
test = np.zeros((5060, 6))

for word_dim in word_dim_list: 
    print('word dim=', word_dim)
    dm =DataManager()
    voc=Vocabulary()
    dm.word_dim=word_dim
    dm.word_len=max_word_len

    voc.word2vec('data/w2v_model/w2v_model_{}'.format(word_dim))
    print("reading data...",end='')
    dm.read_test_data('data/testing_data.csv','test_question','test_option')
    print("\rreading data...finish")

    print("construct data...")
    dm.construct_data_seq2seq('test_question',voc,'data/test_question.npy')
    dm.construct_data_seq2seq('test_option',voc,'data/test_option.npy',multi_seq=True)
    print("construct data...finish")
    print('test_question_seq.shape: '+str(dm.data['test_question'].shape))
    print('test_option.shape: '+str(dm.data['test_option'].shape))

    test = dm.output(dm.data['test_question'])
    test_y = np.argmax(test, axis=1)
    dm.write(test_y, 'ans_{}.csv'.format(word_dim))


