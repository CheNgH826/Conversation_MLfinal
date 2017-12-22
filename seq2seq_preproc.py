import numpy as np
from gen_dict import gen_vocab_list
from keras.utils import to_categorical
import jieba

corpus = 'data/training_data/1_train.txt'
MAX_SENTENCE_LEN = 20

with open(corpus) as f:
    lines = list(f)

word_dict, word_list = gen_vocab_list('vocab.txt')

word_idx_sens = []
for i, line in enumerate(lines):
    word_idx_1sen = [word_dict['<SOS>']]
    seg_list = jieba.lcut(line)
    for w in seg_list:
        if w in word_dict:
            word_idx_1sen.append(word_dict[w])
        else:
            word_idx_1sen.append(word_dict['<unk>'])
    word_idx_1sen.append(word_dict['<EOS>'])
    if len(word_idx_1sen) < MAX_SENTENCE_LEN:
        word_idx_1sen += [word_dict['<blank>']]*(MAX_SENTENCE_LEN-len(word_idx_1sen))
    word_idx_sens.append(to_categorical(word_idx_1sen, len(word_list)))

print(len(word_idx_sens))
print(np.array(word_idx_sens))
np.save('idxed_sentence', np.array(word_idx_sens))

