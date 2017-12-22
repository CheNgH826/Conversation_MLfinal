import jieba
from gensim.models import Word2Vec
import numpy as np
from scipy import spatial

class   Vocabulary:
    def __init__(self):
        pass
    def word2vec(self,path):
        tmp= Word2Vec.load(path)
        self.W2V=tmp
class   DataManager:
    def __init__(self):
        pass
    def read_data(self,path):
        with open(path, 'r') as f:
            next(f)
            lines = list(f)
        questions = []
        options = []
        for i, line in enumerate(lines):
            _, question_i, options_i = line.split(',')
            options_i = options_i.split(':')[1:]
            questions.append(question_i.replace('A:', '').replace('B:', '').replace('C:', '').replace('D:', ''))
            options_i = [opt.replace('\t','').replace('\n','').replace('A', '').replace('B', '').replace('C', '').replace('D', '') for opt in options_i]
            options.append(options_i)
        self.questions=questions
        self.options=options
        #print(questions[:2])
        #print(options[:2])
    def construct_data(self,voc):
        q_vec = []
        for q in self.questions:
            seg_q = list(jieba.cut(q))
            q_vec_list = []
            for w in seg_q:
                if w in voc.W2V:
                    q_vec_list.append(voc.W2V[w])
            q_vec.append(q_vec_list)
        self.questions=q_vec

        opt_vec = []
        for opts_1Q in self.options:
            opt_vec_1Q = []
            for opt in opts_1Q:
                seg_opt = list(jieba.cut(opt))
                opt_vec_list = []
                for w in seg_opt:
                    if w in voc.W2V:
                        opt_vec_list.append(voc.W2V[w])
                opt_vec_1Q.append(opt_vec_list)
            opt_vec.append(opt_vec_1Q)
        self.options=opt_vec
    def caculate_mean(self):
        q_vec_mean = []
        for vec_list in self.q_vec:
            vec_list = np.array(vec_list)
            q_vec_mean.append(np.mean(vec_list,axis=0))
        self.questions_mean=q_vec_mean
        #q_vec_mean = np.array(q_vec_mean)
        opt_vec_mean = []
        for i, opts in enumerate(self.opt_vec):
            means_for_options = []
            for opt in opts:
                opt = np.array(opt)
                means_for_options.append(np.mean(opt,axis=0))
            opt_vec_mean.append(means_for_options)
        #opt_vec_mean = np.array(opt_vec_mean)
        self.options_mean=opt_vec_mean

    def predict(self):
        ans = []
        for i, qmean in enumerate(self.questions_mean):
            sim_6 = []
            for opt in self.questions_mean[i]:
                sim_1 = 1-spatial.distance.cosine(qmean, opt)
                sim_6.append(sim_1)
            ans.append(np.argmax(sim_6))
        self.answer=ans
        print(len(ans))
        print(ans[:10])
