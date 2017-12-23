
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
from keras.callbacks import ModelCheckpoint,EarlyStopping
assert jieba and np 

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
n_batch=1024
adam=keras.optimizers.Adam(clipnorm=0.0001)
adamax=keras.optimizers.Adamax(clipnorm=0.0001)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       create model                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm =DataManager()
voc=Vocabulary()

voc.word2vec('data/w2v_model')

print("reading data...",end='')
dm.read_train_data('data/training_data/1_train.txt','train1')
dm.read_train_data('data/training_data/2_train.txt','train2')
dm.read_train_data('data/training_data/3_train.txt','train3')
dm.read_train_data('data/training_data/4_train.txt','train4')
dm.read_train_data('data/training_data/5_train.txt','train5')
dm.read_test_data('data/testing_data.csv','test_question','test_option')
print("\rreading data...finish")
print(dm.data['train1'][:3])
print(dm.data['train2'][:3])
print(dm.data['train3'][:3])
print(dm.data['train4'][:3])
print(dm.data['train5'][:3])

print("construct data...",end='')
dm.construct_data('train1',voc,'data/train1.npy')
dm.construct_data('train2',voc,'data/train2.npy')
dm.construct_data('train3',voc,'data/train3.npy')
dm.construct_data('train4',voc,'data/train4.npy')
dm.construct_data('train5',voc,'data/train5.npy')
dm.construct_data('test_question',voc,'data/test_question.npy')
dm.construct_data('test_option',voc,'data/test_option.npy',multi_seq=True)
print("\rconstruct data...finish")

model=dm.construct()

print(dm.data['test_option'].shape)
print(dm.data['test_option'][0,1,:,:].shape)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       compile model                            '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.compile(optimizer=adam,loss={'main_output':'cosine_proximity'},metrics=['accuracy'])
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting checkpoint                       '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
filepath="weights/weights.hdf5"
checkpoint1= ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint2=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
callbacks_list = [checkpoint1,checkpoint2]
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       print model                              '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.summary()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       fit model                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
train_x=np.concatenate((dm.data['train1'][:-1],dm.data['train2'][:-1],dm.data['train3'][:-1],dm.data['train4'][:-1],dm.data['train5'][:-1]))
train_y=np.concatenate((dm.data['train1'][1:],dm.data['train2'][1:],dm.data['train3'][1:],dm.data['train4'][1:],dm.data['train5'][1:]))
print(train_x.shape)
print(train_y.shape)
model.fit({'sequence_in':train_x},{'main_output':train_y},epochs=100,batch_size=n_batch,shuffle=True,validation_split=0.05,callbacks=callbacks_list,verbose=1)
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       save model                               '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.save('model.hdf5')

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       writing output                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
test_y=model.predict({'sequence_in':dm.data['test_question']},batch_size=n_batch, verbose=1)
output=dm.output(test_y)
dm.write(output,'./output.csv')
