import numpy as np
import json
import argparse
import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

def calTFIDF(text):
    vectorizer = CountVectorizer(lowercase=True)
    wordcount = vectorizer.fit_transform(text)
    tf_idf_transformer = TfidfTransformer()
    tfidf_matrix = tf_idf_transformer.fit_transform(wordcount)
    return vectorizer, tfidf_matrix




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/mnt/data/mjs/ztl/HSG_myself/cnndm/train.label.jsonl', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='cnndm', help='dataset name')
    
    args = parser.parse_args()

    save_dir = os.path.join('cache',args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir,'filter_word.txt')

    # text = []
    all_sent = []
    cnt = 1
    with open(args.data_path,encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line) # dict
            text = item['text']
            all_sent.extend(text)

    f.close()

    vectorizer,tf_idf_matrix = calTFIDF(all_sent)
    id2word = vectorizer.get_feature_names()

    # 其中每一行对应于一个文档，每一列对应一个单词。mean（0）计算每列的平均值，表示所有文档中每个单词的平均TF-IDF分数。得到的tf_idf变量是一个1D numpy数组
    tf_idf = np.array(tf_idf_matrix.mean(0)) # mean(0) 意味着平均来看每个position上的tfidf值,tf_idf -> 1 x sent_num
    word_order = np.argsort(tf_idf)
    f = open(save_file,'w')
    for idx in word_order[0]:
        # print(idx)
        word = id2word[idx]
        f.write(word)
        f.write("\n")
    f.close()


main()
            

