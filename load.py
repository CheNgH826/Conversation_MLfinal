import numpy as np
import sys

train1 = []
with open(sys.argv[1], 'r', encoding='UTF-8') as fp:
	for i in range (17000):
		a = fp.readline().replace('\n','').split('\n')
		train1.append(a)
train1 = np.array(train1) 

train2 = []
with open(sys.argv[2], 'r', encoding='UTF-8') as fp:
	for i in range (24000):
		a = fp.readline().replace('\n','').split('\n')
		train2.append(a)
train2 = np.array(train2) 

train3 = []
with open(sys.argv[3], 'r', encoding='UTF-8') as fp:
	for i in range (36000):
		a = fp.readline().replace('\n','').split('\n')
		train3.append(a)
train3 = np.array(train3) 

train4 = []
with open(sys.argv[4], 'r', encoding='UTF-8') as fp:
	for i in range (600000):
		a = fp.readline().replace('\n','').split('\n')
		train4.append(a)
train4 = np.array(train4) 

train5 = []
with open(sys.argv[5], 'r', encoding='UTF-8') as fp:
	for i in range (80000):
		a = fp.readline().replace('\n','').split('\n')
		train5.append(a)
train5 = np.array(train5) 

train1 = np.array(train1)
train2 = np.array(train2)
train3 = np.array(train3)
train4 = np.array(train4)
train5 = np.array(train5)




print('train1.shape = ',train1.shape)
print('train1 = ',train1)

print('train2.shape = ',train2.shape)
print('train2 = ',train2)

print('train3.shape = ',train3.shape)
print('train3 = ',train3)

print('train4.shape = ',train4.shape)
print('train4 = ',train4)

print('train5.shape = ',train5.shape)
print('train5 = ',train5)

np.save('train1',train1)
np.save('train2',train2)
np.save('train3',train3)
np.save('train4',train4)
np.save('train5',train5)


