import os
import argparse
import json

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def GetType(path):
    filename = path.split("/")[-1]
    return filename.split(".")[0]


def calTFIDF(text):
    vectorizer = CountVectorizer(lowercase=True)
    wordcount = vectorizer.fit_transform(text)
    tf_idf_transformer = TfidfTransformer()
    tfidf_matrix = tf_idf_transformer.fit_transform(wordcount)
    return vectorizer, tfidf_matrix


def createTfidfDict(matrix, id2word):
    res = dict()
    for i in range(len(matrix)):
        res[i] = dict()
        for j in range(len(matrix[i])):
            weight = matrix[i][j]
            if weight != 0:
                res[i][id2word[j]] = weight

    return res


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        type=str,
        default='/mnt/data/mjs/ztl/HSG_myself/cnndm/train_small.label.jsonl',
        help='file to deal with')
    parser.add_argument('--dataset',
                        type=str,
                        default='cnndm',
                        help='dataset name')

    args = parser.parse_args()

    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fname = GetType(args.data_path) + ".w2s.tfidf.jsonl"
    saveFile = os.path.join(save_dir, fname)
    print("Save word2sent features of dataset %s to %s" %
          (args.dataset, saveFile))

    all_sent = []
    fout = open(saveFile, 'w')
    with open(args.data_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)  # dict
            text = item['text']
            # all_sent.extend(text)
            vectorizer, tf_idf_matrix = calTFIDF(text)
            id2word = vectorizer.get_feature_names()
            tf_idf_matrix = tf_idf_matrix.toarray()
            tf_idf_dict = createTfidfDict(tf_idf_matrix, id2word)
            fout.write(json.dumps(tf_idf_dict) + "\n")

    f.close()
    # print(all_sent)

    fout.close()


main()
