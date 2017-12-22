import numpy as np
import jieba
from gensim.models import Word2Vec

train1 = np.load('train1.npy')
train2 = np.load('train2.npy')
train3 = np.load('train3.npy')
train4 = np.load('train4.npy')
train5 = np.load('train5.npy')

padding1 = []
padding2 = []
padding3 = []
padding4 = []
padding5 = []
 
model = Word2Vec.load('w2v_model')

pad = [ 0 for i in range(100)] 


def process(x,arr):
	for i in range(x.shape[0]):
		tmp = []
		seq = list(jieba.cut(x[i][0]))
		for word in seq:
			if word in model.wv.vocab:
				tmp.append(model[word])
		if len(tmp)<13:
			for i in range(13-len(tmp)):
				tmp.append(pad)
		arr.append(tmp)


		
process(train1,padding1)
padding1 = np.array(padding1)
np.save('padding1',padding1)

process(train2,padding2)
padding2 = np.array(padding2)
np.save('padding2',padding2)

process(train3,padding3)
padding3 = np.array(padding3)
np.save('padding3',padding3)

process(train4,padding4)
padding4 = np.array(padding4)
np.save('padding4',padding4)

process(train5,padding5)
padding5 = np.array(padding5)
np.save('padding5',padding5)


