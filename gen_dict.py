import pandas as pd
import numpy as np
import constants
import jieba
import os


nb_vocab = constants.VOCAB_LIST_SIZE

_PAD = constants.PAD_WORD
_EOS = constants.EOS_WORD
_UNK = constants.UNK_WORD
_SOS = constants.SOS_WORD

PAD_ID = constants.PAD
EOS_ID = constants.EOS
UNK_ID = constants.UNK
SOS_ID = constants.SOS

_START_VOCAB = [_PAD,_SOS,_UNK,_EOS]



def gen_vocab_list(vocab_path,data=None,rewrite = False):
	if rewrite:
		print("Creating vocabulary list with size of %d"%(nb_vocab))
		vocab_dict = dict()
		cut = [jieba.lcut(line) for line in data]
		for item in cut:
			for word in item:
				if word not in vocab_dict:
					vocab_dict[word] = 1
				else:
					vocab_dict[word] += 1
			

		vocab_list = _START_VOCAB + sorted(vocab_dict, key = vocab_dict.get ,reverse = True)
		print (len(vocab_list))
		with open('word_frequency.txt','w') as f:
			for word in vocab_list:
				if word in vocab_dict:
					f.write(word+' '+str(vocab_dict[word])+'\n')
				else:
					f.write(word+' 1'+'\n')

		if len(vocab_list) > nb_vocab:
			vocab_list = vocab_list[:nb_vocab]
		with open(vocab_path,'w') as f:
			for word in vocab_list:
				f.write(word+'\n')	
	else:
		vocab_list = []
		with open(vocab_path) as f:
			vocab_list.extend(f.readlines())
		vocab_list = [line.strip() for line in vocab_list]
	vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])
	return vocab, vocab_list

def load_data(filepath):
	headers = ["text"]
	data = pd.read_csv(filepath,sep = 'delimiter',engine = 'python',names = headers)
	return data['text']

def preprocess(data,vocab_dict):
	
	data = [pad_sentence(line,vocab_dict) for line in data] 
	data = [(data[i],data[i+1]) for i in range(len(data)-1)]
	return (data)

def pad_sentence(sentence,vocab_dict):
	seglist = jieba.lcut(sentence)
	sentence = [vocab_dict.get(w,UNK_ID) for w in seglist]
	#sentence.insert(0,1) # insert start id
	if len(sentence) > 9:
		sentence = sentence[:9]

	# not neccesary
	#sentence.extend([PAD_ID]*(4 -len(sentence))) # (data,10)
	sentence.append(3) # insert end of string id
	#print (sentence)
	return (sentence)
	#return sentence
	




if __name__ == '__main__':
	d = load_data('data/training_data/1_train.txt')
	e = load_data('data/training_data/2_train.txt')
	f = load_data('data/training_data/3_train.txt')
	g = load_data('data/training_data/4_train.txt')
	h = load_data('data/training_data/5_train.txt')

	a = list(d)+list(e)+list(f)+list(g)+list(h)
	
	#print (len(e))
	#print (len(f))
	#print (len(g))
	#print (len(h))
	#print (len(a))

	m,n = gen_vocab_list('vocab.txt',a,rewrite = False)
	
	sentence = pad_sentence(d[0],m)
	print ("Example pad")
	print (sentence)
	#sentence = np.array(to_categorical(sentence,num_classes = 10000))
	print (np.array(sentence).shape)
	#print (n[4766],n[17],n[5052])
	#preprocess(d,m)
	#print (m)
