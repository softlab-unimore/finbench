#!/usr/bin/env python
# pylint: disable=W0201
import argparse

# torch
import torch

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import import_class

from processor.io import IO


class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_data()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_optimizer()

    def init_environment(self):

        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        pass

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(self.arg, flag='train'),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker * torchlight.ngpu(
                self.arg.device),
            drop_last=True)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(self.arg, flag='test'),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker * torchlight.ngpu(
                self.arg.device))
        self.data_loader['valid'] = torch.utils.data.DataLoader(
            dataset=Feeder(self.arg, flag='valid'),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker * torchlight.ngpu(
                self.arg.device))

        self.model_args['graph_args']['tickers'] = self.data_loader['train'].dataset.tickers

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.epoch_info['MAE'] = 0
        self.show_epoch_info()

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.best_performance = {'mean_loss':99999, 'mrr':0, 'top1':0, 'top5':0, 'top10':0}
        # training phase
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()

        self.io.print_log('Model:   {}.'.format(self.arg.model))

        # evaluation
        self.io.print_log('Evaluation Start:')
        self.test()

    @staticmethod
    def get_parser(add_help=False):
        parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

        # data
        parser.add_argument('--data_path', type=str, default='../../Evaluation/data', help='Path to the dataset')
        parser.add_argument('--universe', type=str, default='sp500', help='Universe of stocks to use')
        parser.add_argument('--model_name', type=str, default='RT-GCN', help='Name of the model to use')
        parser.add_argument('--config', default=None, help='path to the configuration file')
        parser.add_argument('--work_dir', default='./work_dir/tmp', help='the work folder for storing results')

        parser.add_argument('--pred_len', type=int, default=5, help='Steps for future prediction')
        parser.add_argument('--seq_len', type=int, default=4, help='Lookback length for the model')
        parser.add_argument('--start_date', type=str, default='2019-01-01', help='Start date for the dataset')
        parser.add_argument('--end_train_date', type=str, default='2021-12-31', help='End date for training set')
        parser.add_argument('--start_valid_date', type=str, default='2022-01-01', help='Start date for validation set')
        parser.add_argument('--end_valid_date', type=str, default='2022-12-31', help='End date for validation set')
        parser.add_argument('--start_test_date', type=str, default='2023-01-01', help='Start date for the test set')
        parser.add_argument('--end_date', type=str, default='2023-12-31', help='End date for the dataset')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False, help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=100, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=10, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=20, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=1, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder.Feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=1, help='the number of worker per gpu for data loader')
        parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')
        parser.add_argument('--in_channels', type=int, default=4, help='the number of input image channels')

        # model
        parser.add_argument('--model', default='net.rt_gcn.Model', help='the model will be used')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        parser.add_argument('--dropout', type=float, default=0.5, help='the dropout rate for the model')
        parser.add_argument('--strategy', type=str, default='uniform')
        parser.add_argument('--edge_importance_weighting', type=str, default='Time-aware', help='whether to use edge importance weighting')
        parser.add_argument('--seed', type=int, default=42, help='Random seed of the model')
        parser.add_argument('--alpha', type=float, default=0.1, help='loss weight between classification loss and contrastive loss')

        return parser