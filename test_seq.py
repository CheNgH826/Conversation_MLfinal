
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
import sys 
sys.path.append("./")
from util import DataManager,Vocabulary
import jieba
import numpy as np
import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping
import argparse
assert jieba and np and ModelCheckpoint and EarlyStopping

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
parser = argparse.ArgumentParser(description='Handle input model.')
parser.add_argument('--model', dest='model',type=str,required=True)
args = parser.parse_args()
continue_file=args.model
n_batch=4096
max_word_len=14
word_dim=300

adam=keras.optimizers.Adam(clipnorm=0.0001)
adamax=keras.optimizers.Adamax(clipnorm=0.0001)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       create model                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm =DataManager()
voc=Vocabulary()
dm.word_dim=word_dim
dm.word_len=max_word_len

voc.word2vec('data/w2v_model')

print("reading data...",end='')
dm.read_train_data('data/training_data/1_train.txt','train1')
dm.read_test_data('data/testing_data.csv','test_question','test_option')
print("\rreading data...finish")
print(dm.data['test_question'][:6])

print("construct data...")
dm.construct_data_seq2seq('train1',voc,'data/train1_seq.npy')
dm.construct_data_seq2seq('test_question',voc,'data/test_question_seq.npy')
dm.construct_data_seq2seq('test_option',voc,'data/test_option_seq.npy',multi_seq=True)
print("construct data...finish")
print('test_question_seq.shape: '+str(dm.data['test_question'].shape))
print('test_option.shape: '+str(dm.data['test_option'].shape))

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       loading model                            '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model=load_model(continue_file)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       print model                              '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.summary()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       decoder                                  '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
data_in=dm.wrape_encoder(dm.data['test_question'],voc)
print(data_in.shape)
test_model=dm.construct_seq2seq_test(model,1024)
data_out=[]
for i in range(len(data_in)):
    print('\rdecoding... sequence: '+str(i),end='')
    data_out.append(dm.decode_seq(data_in[i].reshape((1,14,300)),test_model,voc))
data_out=np.array(data_out)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       writing output                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
'''
output=dm.output(data_in)
dm.write(output,'./output_seq2seq.csv')
'''
'''
