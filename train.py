
import sys 
sys.path.append("./")
from util import DataManager,Vocabulary
import jieba
import numpy as np
assert jieba and np 

test_file = 'data/testing_data.csv'
w2v_model = 'data/w2v.mdl'

dm =DataManager()
voc=Vocabulary()

voc.word2vec(w2v_model)
dm.read_data(test_file)
dm.construct_data(voc)
print(dm.questions)
print(dm.options)
