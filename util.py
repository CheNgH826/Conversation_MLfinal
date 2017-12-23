import jieba
from gensim.models import Word2Vec
import numpy as np
import os.path
from keras.layers import Input, Embedding, LSTM, Dense,Dot, Flatten, Add
from keras.layers import BatchNormalization,Dropout,GRU
from keras.layers import Bidirectional,TimeDistributed
from keras.models import Model
from scipy.spatial.distance import cosine
import pandas as pd
assert np
assert Input and Embedding and LSTM and Dense and Dot and Flatten and Add
assert Model and BatchNormalization and Dropout and GRU

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
        self.word_len=13
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
    def construct_data(self,name,voc,outputfile,multi_seq=False):
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
    def wrape_decoder(self,data,voc,decode_in=True):
        res=[]
        for i in range(len(data)):
            print('\rprocessing sequence number: '+str(i),end='')
            if (len(data[i])==0): continue
            if (decode_in):
                data[i]=np.array(data[i])
                res.append(np.concatenate((voc.W2V['<BOS>'].reshape((1,-1)),data[i]),axis=0))
            else:
                data[i]=np.array(data[i])
                res.append(np.concatenate((data[i],voc.W2V['<EOS>'].reshape((1,-1))),axis=0))
        print('\rprocess finish...')
        return np.array(res)
    def construct_LSTM(self,unit=128):
        sequence_in= Input(shape=(self.word_len,self.word_dim), name='sequence_in')
        x=Bidirectional(LSTM(unit//2,activation='sigmoid',return_sequences=True,init='glorot_normal',inner_init='glorot_normal'))(sequence_in)
        for i in range(3):
            x=Bidirectional(LSTM(unit//2,activation='sigmoid',return_sequences=True,init='glorot_normal',inner_init='glorot_normal'))(x)
        x=TimeDistributed(Dense(unit,activation='relu'))(x)
        main_output=TimeDistributed(Dense(self.word_dim,activation='softmax'),name='main_output')(x)
        model=Model(inputs=sequence_in,outputs=main_output)
        return model
    def construct_seq2seq(self,unit=128):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.word_dim))
        encoder = LSTM(unit, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.word_dim))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the 
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(unit, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
        decoder_dense = Dense(self.word_dim, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return model
    def average(self,data):
        return np.mean(data,axis=0)
    def output(self,data):
        answer=[]
        for i in range(len(data)):
            ans=self.average(data[i])
            opt=[self.average(self.data['test_option'][i,j,:,:]) for j in range(6)]
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

