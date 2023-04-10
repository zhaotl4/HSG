import numpy as np
from tools.logger import *

class Word_Embedding(object):
    def __init__(self,path,vocab):
        logger.info("[INFO] Loading the external word embedding")
        self.path = path
        self.vocab_list = vocab.word_list()
        self.vocab = vocab

    def load_my_vecs(self,k):

        word_dict = {}

        with open(self.path,encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(" ")
                word = line[0]
                vector = []
                if word in self.vocab_list:
                    for i in range(1,len(line)):
                        if i > k:
                            break
                        vector.append(float(line[i]))
                    word_dict[word] = vector
        
        f.close()
        return word_dict

    def add_unknown_words_by_avg(self,word_dict,k):

        word_dict_list = []
        
        for word in self.vocab_list:
            if word in word_dict:
                word_dict_list.append(word_dict[word])

        avg_words = []
        total_cnt = len(word_dict_list)
        for i in range(k):
            avg_reprsent = 0.0
            for vec in word_dict_list:
                avg_reprsent += vec[i]
            avg_reprsent = round(avg_reprsent,6) # 四舍五入
            avg_words.append(float(avg_reprsent/total_cnt))

        word2vec_list = []
        for i in range(self.vocab.size()):
            word = self.vocab.id2word(i)
            if word not in word_dict.keys():
                word_dict[word] = avg_words

            word2vec_list.append(word_dict[word])
        
        return word2vec_list



        


        

                

