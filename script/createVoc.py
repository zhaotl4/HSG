import json
import os
import argparse
import nltk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='/mnt/data/mjs/ztl/HSG_myself/cnndm/train.label.jsonl',
        help='file to deal with')
    parser.add_argument('--dataset',
                        type=str,
                        default='cnndm',
                        help='dataset name')

    args = parser.parse_args()

    save_dir = os.path.join('cache', args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, 'vocab')
    print('Save vocab of ', args.dataset, 'to ', save_file)

    text = []
    label = []
    allWord = []
    cnt = 0

    print('Begin calculate vocab')

    with open(args.data_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            cnt += 1
            text = ' '.join(item['text'])
            # print(text)
            allWord.extend(text.split())
            # print(item) # item is a dict
            # if cnt > 1:
            #     break
    f.close()

    # print(allWord)
    fdist1 = nltk.FreqDist(allWord)
    keys = fdist1.most_common()
    # print(keys)
    # if not os.path.exists(save_file):
    f = open(save_file, 'w')
    # with open(save_file,encoding='utf-8') as f:
    for key in keys:
        line = str(key[0]) + ' ' + str(key[1])
        f.write(line)
        f.write("\n")
    f.close()


main()
