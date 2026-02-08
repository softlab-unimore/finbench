#!/usr/bin/env python
# pylint: disable=W0201
import json
import os
import pickle
import time
import argparse
import numpy as np

# torch
import torch
import torch.optim as optim

from feeder.tools import get_metrics
# loss
from net.loss import Loss

# torchlight
from torchlight import str2bool

from processor.processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,**(self.model_args))
        self.model.apply(weights_init)
        # self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.MSELoss()
        self.loss = Loss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            # self.lr = self.arg.base_lr
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_topk(self, k):
        rank = self.result.argsort() # self.result shape (batch_size, num_node)
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_MRR(self):
        # calculate the MRR of the top 1 return-ration stock
        mrr_top = 0.0
        for i in range(self.label.shape[0]) :
            top1_pos_in_gt = 0
            rank_gt = np.argsort(self.label[i])
            rank_pre = np.argsort(self.result[i])
            pre_top1 = set()
            for j in range(1, self.label.shape[1] + 1):
                cur_rank_pre = rank_pre[-1 * j]    # chose the max return-ration stock' id in grouth truth
                cur_rank_gt = rank_gt[-1 * j]
                if len(pre_top1) < 1:
                    pre_top1.add(cur_rank_pre)
                top1_pos_in_gt += 1
                if cur_rank_gt in pre_top1:
                    break
            mrr_top += 1.0 / top1_pos_in_gt
        self.mrrt = mrr_top / self.result.shape[0]

    def show_return_ration(self, k):
        # calculate the return-ration of the top k stocks
        self.bt_k = 1.0
        for i in range(self.result.shape[0]):
            rank_pre = np.argsort(self.result[i])

            pre_topk = set()
            for j in range(1, self.result.shape[1] + 1):
                cur_rank = rank_pre[-1 * j]
                if len(pre_topk) < k:
                    pre_topk.add(cur_rank)

            # back testing on top k
            return_ration_topk = 0
            for index in pre_topk:
                return_ration_topk += self.label[i][index]
            return_ration_topk /= k
            with open('IRR_{}.txt'.format(k),'a+') as f:
                f.write(str(round((return_ration_topk),2)))
                f.write('\n')
                f.close()
            self.bt_k += return_ration_topk

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        start_time = time.time()
        for data, closing_price, label in loader:

            data = data.float().to(self.dev)
            closing_price = closing_price.float().to(self.dev)
            label = label.float().to(self.dev)

            # forward
            output = self.model(data)
            prediction = torch.div(torch.sub(output, closing_price), closing_price)
            loss = self.loss(prediction, label, self.arg.alpha)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_log('Time consumption:{:.4f}s'.format(time.time()-start_time))

    def test(self, evaluation=True):

        self.model.eval()
        loader_test = self.data_loader['test']
        loader_valid = self.data_loader['valid']

        loss_value = []
        result_frag = []
        label_frag = []

        for data, closing_price, label in loader_valid:

            # get data
            data = data.float().to(self.dev)
            closing_price = closing_price.float().to(self.dev)
            label = label.float().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
                prediction = torch.div(torch.sub(output, closing_price), closing_price)
            result_frag.append(prediction.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(prediction, label, self.arg.alpha)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        label_frag = np.concatenate(label_frag, axis=0)
        result_frag = np.concatenate(result_frag, axis=0)

        metrics_path = f'./results/{self.arg.universe}/{self.arg.model_name}/{self.arg.seed}/y{self.arg.start_test_date.split("-")[0]}'
        os.makedirs(metrics_path, exist_ok=True)

        valid_metrics = get_metrics(label_frag, result_frag)
        valid_metrics = {k: float(v) for k, v in valid_metrics.items()}

        with open(f'{metrics_path}/val_metrics_sl{self.arg.seq_len}_pl{self.arg.pred_len}.json', 'w') as f:
            json.dump(valid_metrics, f, indent=4)

        loss_value = []
        result_frag = []
        label_frag = []

        for data, closing_price, label in loader_test:
            
            # get data
            data = data.float().to(self.dev)
            closing_price = closing_price.float().to(self.dev)
            label = label.float().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
                prediction = torch.div(torch.sub(output, closing_price), closing_price)
            result_frag.append(prediction.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(prediction, label, self.arg.alpha)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        label_frag = np.concatenate(label_frag, axis=0)
        result_frag = np.concatenate(result_frag, axis=0)

        test_metrics = get_metrics(label_frag, result_frag)
        test_metrics = {k: float(v) for k, v in test_metrics.items()}

        label_frag = [np.expand_dims(label_frag[i], axis=1) for i in range(label_frag.shape[0])]
        result_frag = [np.expand_dims(result_frag[i], axis=1) for i in range(result_frag.shape[0])]

        dataset = self.data_loader['test'].dataset

        results = {
            'metrics': test_metrics,
            'preds': result_frag,
            'labels': label_frag,
            'pred_date': dataset.dates_gt.to_list(),
            'last_date': dataset.last_seq_date.to_list(),
            'tickers': [dataset.tickers]*len(result_frag)
        }

        with open(f'{metrics_path}/metrics_sl{self.arg.seq_len}_pl{self.arg.pred_len}.json', 'w') as f:
            json.dump(test_metrics, f, indent=4)

        with open(f'{metrics_path}/results_sl{self.arg.seq_len}_pl{self.arg.pred_len}.pkl', 'wb') as f:
            pickle.dump(results, f)

        print('Metrics: ', test_metrics)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Relational Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5, 10], nargs='+', help='which Top K return ration will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
