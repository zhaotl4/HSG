import torch
import dgl

import os
from tools.utils import eval_label
from tools.logger import *


class TestPipLine():
    def __init__(self, model, m, test_dir, limited) -> None:
        self.model = model
        self.limited = limited
        self.m = m
        self.test_dir = test_dir
        self.extracts = []

        self.batch_number = []
        self.running_loss = 0
        self.example_num = 0
        self.total_sent_num = 0

        self._hyps = []
        self._refer = []

    def evaluation(self, G, index, valset):
        pass

    def get_metric(self):
        pass

    def save_decode_file(self):
        import datetime
        now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(self.test_dir, now_time)
        with open(log_dir, 'wb') as f:
            for i in range(self.rouge_pair_num):
                f.write(b'[Reference]\t')
                f.write(self._refer[i].encode('utf-8'))
                f.write(b'\n')
                f.write(b'[Hypothesis]\t')
                f.write(self._hyps[i].encode('utf-8'))
                f.write(b'\n')
                f.write(b'\n')
                f.write(b'\n')

    @property
    def running_avg_loss(self):
        return self.running_loss / self.batch_number

    @property
    def rouge_pair_num(self):
        return len(self._hyps)

    @property
    def hyps(self):
        if self.limited:
            hlist = []
            for i in range(self.rouge_pair_num):
                k = len(self._refer[i].split(' '))
                lh = " ".join(self._hyps[i].split(' ')[:k])
                hlist.append(lh)
            return hlist
        else:
            return self._hyps

    @property
    def refer(self):
        return self._refer

    @property
    def extract_label(self):
        return self.extracts


class SLTester(TestPipLine):
    def __init__(self,
                 model,
                 m,
                 test_dir=None,
                 limited=False,
                 blocking_win=3) -> None:
        super().__init__(model, m, test_dir, limited)
        self.pred, self.true, self.match, self.match_true = 0, 0, 0, 0
        self._F = 0
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.blocking_win = blocking_win

    def evaluation(self, G, index, valset, blocking=False):
        self.batch_number += 1
        outputs = self.model.forward(G)
        sent_node_id = G.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
        label = G.ndata['label'][sent_node_id].sum(-1)
        G.nodes[sent_node_id].data['loss'] = self.criterion(
            outputs, label).unsqueeze(-1)
        loss = dgl.sum_nodes(G, 'loss')
        loss = loss.mean()
        self.running_loss += float(loss.data)

        G.nodes[sent_node_id].data['p'] = outputs
        glist = dgl.unbatch(G)
        for j in range(len(glist)):
            idx = index[j]
            example = valset.get_example(idx)
            origin_article_sents = example.origin_article_sents
            sent_max_num = len(origin_article_sents)
            refer = example.original_abstract

            g = glist[j]
            sent_node_id = g.filter_nodes(
                lambda nodes: nodes.data['dtype'] == 1)
            N = len(sent_node_id)
            predict_sent = g.ndata['p'][sent_node_id]
            predict_sent = predict_sent.view(-1, 2)  # [node,2]
            label = g.ndata['label'][sent_node_id].sum(
                -1).squeeze().cpu()  #[n_node]
            if self.m == 0:
                prediction = predict_sent.max(1)[1]  #[node]
                pred_idx = torch.arange(N)[prediction != 0].long()
            else:
                if blocking:
                    pred_idx = self.ngram_blocking(origin_article_sents,
                                                   predict_sent[:, 1],
                                                   self.blocking_win,
                                                   min(self.m, N))
                else:
                    topK, pred_idx = torch.topk(predict_sent[:, 1],
                                                min(self.m, N))
                prediction = torch.zeros(N).long()
                prediction[pred_idx] = 1

            self.extracts.append(pred_idx.tolist())
            self.pred = prediction.sum()
            self.true = label.sum()

            # prediction == 1 is a boolean expression that returns a binary tensor where each element is True if the corresponding element in prediction is 1, and False otherwise.
            self.match_true += ((prediction == label) &
                                (prediction == 1)).sum()
            self.match += (prediction == label).sum()
            self.total_sent_num += N
            self.example_num += 1

            hyps = '\n'.join(origin_article_sents[id] for id in pred_idx
                             if id < sent_max_num)
            self._hyps.append(hyps)
            self._refer.append(refer)

    def get_metric(self):
        logger.info(
            "[INFO] Validset match_true %d, pred %d, true %d, total %d, match %d",
            self.match_true, self.pred, self.true, self.total_sentence_num,
            self.match)
        self._accu, self._precision, self._recall, self._F = eval_label(
            self.match_true, self.pred, self.true, self.total_sentence_num,
            self.match)
        logger.info(
            "[INFO] The size of totalset is %d, sent_number is %d, accu is %f, precision is %f, recall is %f, F is %f",
            self.example_num, self.total_sentence_num, self._accu,
            self._precision, self._recall, self._F)

    def ngram_blocking(self, sents, p_sent, n_win, k):
        ngram_list = []
        _, sorted_idx = p_sent.sort(descending=True)
        S = []
        for idx in sorted_idx:
            sent = sents[idx]
            pieces = sent.split()
            overlap_flag = 0
            sent_ngram = []
            for i in range(len(pieces) - n_win):
                ngram = ' '.join(pieces[i:(i + n_win)])
                if ngram in ngram_list:
                    overlap_flag = 1
                    break
                else:
                    sent_ngram.append(ngram)
                if overlap_flag == 0:
                    S.append(idx)
                    ngram_list.extend(sent_ngram)
                    if len(S) >= k:
                        break

        S = torch.LongTensor(S)

        return S

    @property
    def label_metric(self):
        return self._F