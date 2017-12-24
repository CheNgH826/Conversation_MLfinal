import jieba
from gensim.models import Word2Vec
import numpy as np
import os.path
from keras.layers import Input, Embedding, LSTM, Dense,Dot, Flatten, Add
from keras.layers import BatchNormalization,Dropout,GRU
from keras.layers import Bidirectional,TimeDistributed,Masking
from keras.models import Model
from scipy.spatial.distance import cosine
import pandas as pd
assert np
assert Input and Embedding and LSTM and Dense and Dot and Flatten and Add
assert Model and BatchNormalization and Dropout and GRU and Masking

class   Vocabulary:
    def __init__(self):
        pass
    def word2vec(self,path):
        tmp= Word2Vec.load(path)
        self.W2V=tmp
class   DataManager:
    def __init__(self):
        self.data={}
        self.word_dim=300
        self.word_len=0
    def read_train_data(self,path,name):
        with open(path, 'r') as f:
            lines = list(f)
        data=[]    
        for i in range (len(lines)):
            a = lines[i].replace('\n','')
            data.append(a)
        self.data[name]= np.array(data) 
    def read_test_data(self,path,name_q,name_o):
        with open(path, 'r') as f:
            next(f)
            lines = list(f)
        questions = []
        options = []
        for i, line in enumerate(lines):
            _, question_i, options_i = line.split(',')
            questions.append(question_i.replace('A:', '').replace('B:', '').replace('C:', '').replace('D:', ''))
            options_i = options_i.split(':')[1:]
            options_i = [opt.replace('\t','').replace('\n','').replace('A', '').replace('B', '').replace('C', '').replace('D', '') for opt in options_i]
            options.append(options_i)
        self.data[name_q]=questions
        self.data[name_o]=options
    def construct_data_LSTM(self,name,voc,outputfile,multi_seq=False):
        if(os.path.isfile(outputfile)):
            self.data[name]=np.load(outputfile)
        elif (multi_seq==False):
            vec = []
            for i in self.data[name]:
                seg = list(jieba.cut(i))
                vec_list = []
                for w in seg:
                    if (len(vec_list)>=13):break
                    elif w in voc.W2V:
                        #print(voc.W2V[w].shape)
                        vec_list.append(voc.W2V[w])
                if (len(vec_list)<13):
                    for i in range(13-len(vec_list)):
                        vec_list.append(np.zeros((self.word_dim,)))
                vec.append(vec_list)
            vec=np.array(vec)
            self.data[name]=vec
            print(name,vec.shape)
            np.save(outputfile,vec)
        else:
            opt_vec = []
            for opts_1Q in self.data[name]:
                opt_vec_1Q = []
                for opt in opts_1Q:
                    seg_opt = list(jieba.cut(opt))
                    opt_vec_list = []
                    for w in seg_opt:
                        if (len(opt_vec_list)>=13):break
                        elif w in voc.W2V:
                            opt_vec_list.append(voc.W2V[w])
                    if (len(opt_vec_list)<13):
                        for i in range(13-len(opt_vec_list)):
                            opt_vec_list.append(np.zeros((self.word_dim,)))
                    opt_vec_1Q.append(opt_vec_list)
                opt_vec.append(opt_vec_1Q)
            opt_vec=np.array(opt_vec)
            self.data[name]=opt_vec
            print(name,opt_vec.shape)
            np.save(outputfile,opt_vec)
    def construct_data_seq2seq(self,name,voc,outputfile,multi_seq=False):
        if(os.path.isfile(outputfile)):
            self.data[name]=np.load(outputfile)
        elif (multi_seq==False):
            vec = []
            for i in self.data[name]:
                seg = list(jieba.cut(i))
                vec_list = []
                for w in seg:
                    if w in voc.W2V:
                        vec_list.append(voc.W2V[w])
                vec.append(np.array(vec_list))
            vec=np.array(vec)
            self.data[name]=vec
            print(name,vec.shape)
            np.save(outputfile,vec)
        else:
            vec = []
            for option in self.data[name]:
                opt= []
                for i in option:
                    seg = list(jieba.cut(i))
                    vec_list = []
                    for w in seg:
                        if w in voc.W2V:
                            vec_list.append(voc.W2V[w])
                    opt.append(np.array(vec_list))
                vec.append(np.array(opt))
            vec=np.array(vec)
            self.data[name]=vec
            print(name,vec.shape)
            np.save(outputfile,vec)
    def wrape_encoder(self,data,voc):
        res=[]
        for i in range(len(data)):
            print('\rprocessing sequence number: '+str(i),end='')
            if(len(data[i])==0):
                res.append(np.zeros((self.word_len,self.word_dim)))
            elif(len(data[i])>self.word_len):
                data[i]=np.array(data[i])
                res.append(data[i][:self.word_len,:])
            else:
                data[i]=np.array(data[i])
                res.append(np.concatenate((data[i],np.zeros((self.word_len-len(data[i]),self.word_dim))),axis=0))
        print('\nprocess finish...')
        return np.array(res)
    def wrape_decoder(self,data,voc,decode_in=True):
        res=[]
        for i in range(len(data)):
            print('\rprocessing sequence number: '+str(i),end='')
            if (decode_in):
                if(len(data[i])==0):
                    res.append(np.concatenate((voc.W2V['<BOS>'].reshape((1,-1)),np.zeros((self.word_len-1,self.word_dim))),axis=0))
                elif(len(data[i])>=self.word_len):
                    data[i]=np.array(data[i])
                    res.append(np.concatenate((voc.W2V['<BOS>'].reshape((1,-1)),data[i][:self.word_len-1,:]),axis=0))
                else:
                    data[i]=np.array(data[i])
                    res.append(np.concatenate((voc.W2V['<BOS>'].reshape((1,-1)),data[i],np.zeros((self.word_len-1-len(data[i]),self.word_dim))),axis=0))
            else:
                if(len(data[i])==0):
                    res.append(np.concatenate((np.zeros((self.word_len-1,self.word_dim)),voc.W2V['<EOS>'].reshape((1,-1))),axis=0))
                elif(len(data[i])>=self.word_len):
                    data[i]=np.array(data[i])
                    res.append(np.concatenate((data[i][:self.word_len-1,:],voc.W2V['<EOS>'].reshape((1,-1))),axis=0))
                else:
                    data[i]=np.array(data[i])
                    res.append(np.concatenate((data[i],voc.W2V['<EOS>'].reshape((1,-1)),np.zeros((self.word_len-1-len(data[i]),self.word_dim))),axis=0))
        print('\nprocess finish...')
        return np.array(res)
    def construct_LSTM(self,unit=128):
        sequence_in= Input(shape=(self.word_len,self.word_dim), name='sequence_in')
        x=Bidirectional(LSTM(unit,activation='tanh',return_sequences=True))(sequence_in)
        for i in range(5):
            x=Bidirectional(LSTM(unit,activation='tanh',return_sequences=True))(x)
        x=TimeDistributed(Dense(unit,activation='relu'))(x)
        main_output=TimeDistributed(Dense(self.word_dim,activation='linear'),name='main_output')(x)
        model=Model(inputs=sequence_in,outputs=main_output)
        return model
    def construct_seq2seq_train(self,unit=256):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.word_dim))
        encoder_lstm=LSTM(unit, return_sequences=True)(encoder_inputs)
        encoder =LSTM(unit, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_lstm)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.word_dim))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the 
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(unit, return_sequences=True)(decoder_inputs)
        decoder= LSTM(unit, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder(decoder_lstm,initial_state=encoder_states)
        decoder_outputs= Dense(unit, activation='relu')(decoder_outputs)
        decoder_dense = Dense(self.word_dim, activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return model
    def construct_seq2seq_test(self,model,unit):
        print(model.inputs)
        print(model.outputs)
        print(model.layers)
        encoder_inputs = model.inputs[0] 
        encoder_outputs, state_h, state_c = model.layers[2](encoder_inputs)
        encoder_states = [state_h, state_c]
        encoder_model = Model(encoder_inputs,encoder_states)

        decoder_inputs = Input(shape=(None, self.word_dim))
        decoder_state_input_h = Input(shape=(unit,))
        decoder_state_input_c = Input(shape=(unit,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = model.layers[3](decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = model.layers[4](decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
        return [encoder_model,decoder_model]
    def decode_seq(self,data_in,model,voc):
	# Encode the input as state vectors.
        encoder_model=model[0]
        decoder_model=model[1]
        states_value = encoder_model.predict(data_in)

        # Generate empty target sequence of length 1.
        # Populate the first character of target sequence with the start character.
        target_seq =voc.W2V['<BOS>'].reshape((1,1,-1))

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            #print('output_tokens.shape: ',output_tokens)
            sampled_char= voc.W2V.wv.similar_by_vector(output_tokens[0, -1, :],topn=1)[0][0]
            #print('sampled_char: ',sampled_char)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '<EOS>' or len(decoded_sentence) > self.word_len):
                stop_condition = True
            else:
                decoded_sentence.append(voc.W2V[sampled_char].reshape((1,-1)))

            # Update the target sequence (of length 1).
            target_seq = voc.W2V[sampled_char].reshape((1,1,-1))

            # Update states
            states_value = [h, c]
        return decoded_sentence
    def average(self,data):
        return np.mean(data,axis=0)
    def output(self,data):
        answer=[]
        for i in range(len(data)):
            ans=self.average(data[i])
            opt=[self.average(self.data['test_option'][i,j]) for j in range(6)]
            dist=[1-cosine(ans,opt[i]) for i in range(len(opt))]
            answer.append(dist)
        return np.argmax(np.array(answer),axis=1)
    def write(self,test,path):
        test=test.reshape((-1,1))
        idx=np.array([[j for j in range(1,len(test)+1)]]).T
        test=np.hstack((idx,test))
        #print(test.shape)
        #print(output.shape)
        myoutput=pd.DataFrame(test,columns=["id","ans"])
        myoutput.to_csv(path,index=False)

