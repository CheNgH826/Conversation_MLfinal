from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from gensim.models import Word2Vec

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

##### Data
w2v_mdl = Word2Vec.load('data/w2v.mdl')

train_file_list = ['training_data/'+str(i)+'_train.txt' for i range(1,2)]
vec_of_sen = np.load('data/1_train.npy')
max_sentence_length = 13
num_chinese_vocab = len(w2v_mdl.wv.vocab)

encoder_input_data = vec_of_sen[:-1]
decoder_input_data = vec_of_sen[1:]
decoder_target_data = decoder_target_data[:,]

for train_file in train_file_list:
    with open(train_file) as f:
        sentences = list(f)
    



# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


