
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
print("reading data...",end='')
dm.read_data(test_file)
print("\rreading data...finish")
print("construct data...",end='')
dm.construct_data(voc)
print("\rconstruct data...finish")
