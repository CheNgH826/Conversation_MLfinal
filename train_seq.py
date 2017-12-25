
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
n_epoch=100
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
dm.read_train_data('data/training_data/2_train.txt','train2')
dm.read_train_data('data/training_data/3_train.txt','train3')
dm.read_train_data('data/training_data/4_train.txt','train4')
dm.read_train_data('data/training_data/5_train.txt','train5')
#dm.read_test_data('data/testing_data.csv','test_question','test_option')
print("\rreading data...finish")
print(dm.data['train1'][:3])
print(dm.data['train2'][:3])
print(dm.data['train3'][:3])
print(dm.data['train4'][:3])
print(dm.data['train5'][:3])

print("construct data...")
dm.construct_data_seq2seq('train1',voc,'data/train1_seq.npy')
dm.construct_data_seq2seq('train2',voc,'data/train2_seq.npy')
dm.construct_data_seq2seq('train3',voc,'data/train3_seq.npy')
dm.construct_data_seq2seq('train4',voc,'data/train4_seq.npy')
dm.construct_data_seq2seq('train5',voc,'data/train5_seq.npy')
#dm.construct_data_seq2seq('test_question',voc,'data/test_question_seq.npy')
#dm.construct_data_seq2seq('test_option',voc,'data/test_option_seq.npy',multi_seq=True)
print("construct data...finish")

model=dm.construct_seq2seq_train(1024)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       compile model                            '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.compile(optimizer=adam, loss='cosine_proximity',metrics=['accuracy'])
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
encoder_data=np.concatenate((dm.data['train1'][:-1],dm.data['train2'][:-1],dm.data['train3'][:-1],dm.data['train4'][:-1],dm.data['train5'][:-1]))
decoder_data=np.concatenate((dm.data['train1'][1:],dm.data['train2'][1:],dm.data['train3'][1:],dm.data['train4'][1:],dm.data['train5'][1:]))
encoder_input_data=dm.wrape_encoder(encoder_data,voc)
decoder_input_data=dm.wrape_decoder(decoder_data,voc,decode_in=True)
decoder_target_data=dm.wrape_decoder(decoder_data,voc,decode_in=False)
print('encoder_input_data.shape: ',encoder_input_data.shape)
print('decoder_input_data.shape: ',decoder_input_data.shape)
print('decoder_target_data.shape: ',decoder_target_data.shape)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=n_batch,
          epochs=n_epoch,
          validation_split=0.1,
          shuffle=True,callbacks=callbacks_list,verbose=1)
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       save model                               '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.save('model.hdf5')

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       writing output                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
test_y=model.predict({'sequence_in':dm.data['test_question']},batch_size=n_batch, verbose=1)
output=dm.output(test_y)
dm.write(output,'./output.csv')
'''
