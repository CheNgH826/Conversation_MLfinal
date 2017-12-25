from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from keras.utils import to_categorical
import numpy as np
#from gensim.models import Word2Vec
from gen_dict import gen_vocab_list
import jieba
jieba.dt.cache_file = 'jieba.cache.new'

word_dict, word_list = gen_vocab_list('vocab.txt')
WORD_NUM = len(word_list)
WORDVEC_DIM = 100
latent_dim = 20

# Define an input sequence and process it.
encoder_inputs = Input(shape=(20,))
encoder_embedding = Embedding(WORD_NUM, WORDVEC_DIM, mask_zero=True)(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embedding)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(20,))
decoder_embedding = Embedding(WORD_NUM, WORDVEC_DIM, mask_zero=True)(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding,
                                             initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(WORD_NUM, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

##### Data
#w2v_mdl = Word2Vec.load('data/w2v.mdl')
'''
train_file_list = ['training_data/'+str(i)+'_train.txt' for i range(1,2)]
vec_of_sen = np.load('data/1_train.npy')
max_sentence_length = 13
num_chinese_vocab = len(w2v_mdl.wv.vocab)
'''

idx_sen = np.load('idxed_sentence.npy')
encoder_input_data = idx_sen[:,1:]
decoder_input_data = idx_sen
decoder_target_data = []
decoder_target_data_idx = idx_sen[:,1:]
for sen in decoder_target_data_idx:
    decoder_target_data.append(to_categorical(sen, WORD_NUM))
decoder_target_data = np.array(decoder_target_data)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


