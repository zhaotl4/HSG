'''
需要的是生成两个,word to id, id to word 这两个dict
'''
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UKN]'
START_TOKEN = '[START]'
STOP_TOKEN = '[STOP]'


class Vocab(object):
    def __init__(self, vocab_path, max_size) -> None:
        self.vocab_path = vocab_path
        self.max_size = max_size
        self.cnt = 0
        self.word2id = dict()
        self.id2word = dict()

        for word in [PAD_TOKEN, UNKNOWN_TOKEN, START_TOKEN, STOP_TOKEN]:
            self.word2id[word] = self.cnt
            self.id2word[self.cnt] = word
            self.cnt += 1

        with open(self.vocab_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                word = line[0]
                self.word2id[word] = self.cnt
                self.id2word[self.cnt] = word
                self.cnt += 1
                if self.cnt > self.max_size:
                    break

        f.close()

    def id2word(self, id):
        if id not in self.id2word.keys():
            print('ID NOT FOUND')
            return "UNKNOWN WORD"
        return self.id2word[id]

    def word2id(self, word):
        if word not in self.word2id.keys():
            print('WORD NOT FOUND')
            return self.word2id(UNKNOWN_TOKEN)
        return self.word2id(word)

    def vocab_size(self):
        return self.cnt

    def word_list(self):
        return self.word2id.keys()
