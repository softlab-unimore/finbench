import argparse
import copy
import json
import pickle
import warnings

import numpy as np
import os
import random

from sklearn.utils import shuffle

warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from time import time

from load_dataset import load_dataset
from evaluator import evaluate

class AWLSTM:
    def __init__(self, data_path, universe, model_path, model_save_path, parameters, steps=1, epochs=50,
                 batch_size=256, gpu=False, tra_date='2014-01-02',
                 val_date='2015-08-03', tes_date='2015-10-01', att=0, hinge=0,
                 fix_init=0, adv=0, reload=0, start_date=None, pred_len=1, metrics_path='./results'):
        self.data_path = data_path
        self.universe = universe
        self.model_path = model_path
        self.model_save_path = model_save_path
        self.metrics_path = metrics_path
        # model parameters
        self.paras = copy.copy(parameters)
        # training parameters
        self.steps = steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu

        if att == 1:
            self.att = True
        else:
            self.att = False
        if hinge == 1:
            self.hinge = True
        else:
            self.hinge = False
        if fix_init == 1:
            self.fix_init = True
        else:
            self.fix_init = False
        if adv == 1:
            self.adv_train = True
        else:
            self.adv_train = False
        if reload == 1:
            self.reload = True
        else:
            self.reload = False

        # load data
        self.tra_date = tra_date
        self.val_date = val_date
        self.tes_date = tes_date
        self.pred_len = pred_len

        self.tra_pv, self.tra_wd, self.tra_gt, self.val_pv, self.val_wd, self.val_gt, self.tes_pv, self.tes_wd, self.tes_gt, self.test_tickers, self.test_last_dates, self.test_dates = load_dataset(
            self.data_path, self.universe, start_date, tes_date, tra_date, val_date, self.paras['seq'], pred_len
        )

        self.fea_dim = self.tra_pv.shape[2]

    def get_batch(self, sta_ind=None):
        if sta_ind is None:
            sta_ind = random.randrange(0, self.tra_pv.shape[0])
        if sta_ind + self.batch_size < self.tra_pv.shape[0]:
            end_ind = sta_ind + self.batch_size
        else:
            sta_ind = self.tra_pv.shape[0] - self.batch_size
            end_ind = self.tra_pv.shape[0]
        return self.tra_pv[sta_ind:end_ind, :, :], \
               self.tra_wd[sta_ind:end_ind, :, :], \
               self.tra_gt[sta_ind:end_ind, :]

    def adv_part(self, adv_inputs):
        print('adversial part')
        if self.att:
            with tf.variable_scope('pre_fc'):
                self.fc_W = tf.get_variable(
                    'weights', dtype=tf.float32,
                    shape=[self.paras['unit'] * 2, 1],
                    initializer=tf.glorot_uniform_initializer()
                )
                self.fc_b = tf.get_variable(
                    'biases', dtype=tf.float32,
                    shape=[1, ],
                    initializer=tf.zeros_initializer()
                )
                if self.hinge:
                    pred = tf.nn.bias_add(
                        tf.matmul(adv_inputs, self.fc_W), self.fc_b
                    )
                else:
                    pred = tf.nn.sigmoid(
                        tf.nn.bias_add(tf.matmul(self.fea_con, self.fc_W),
                                       self.fc_b)
                    )
        else:
            # One hidden layer
            if self.hinge:
                pred = tf.layers.dense(
                    adv_inputs, units=1, activation=None,
                    name='pre_fc',
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
            else:
                pred = tf.layers.dense(
                    adv_inputs, units=1, activation=tf.nn.sigmoid,
                    name='pre_fc',
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
        return pred

    def construct_graph(self):
        print('is pred_lstm')
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()
            if self.fix_init:
                # tf.set_random_seed(123456)
                tf.set_random_seed(self.fix_init)

            self.gt_var = tf.placeholder(tf.float32, [None, 1])
            self.pv_var = tf.placeholder(tf.float32, [None, self.paras['seq'], self.fea_dim])
            self.wd_var = tf.placeholder(tf.float32, [None, self.paras['seq'], 5])

            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.paras['unit']
            )

            self.in_lat = tf.layers.dense(
                self.pv_var, units=self.fea_dim,
                activation=tf.nn.tanh, name='in_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            self.outputs, _ = tf.nn.dynamic_rnn(
                # self.outputs, _ = tf.nn.static_rnn(
                self.lstm_cell, self.in_lat, dtype=tf.float32
                # , initial_state=ini_sta
            )

            self.loss = 0
            self.adv_loss = 0
            self.l2_norm = 0
            if self.att:
                with tf.variable_scope('lstm_att') as scope:
                    self.av_W = tf.get_variable(
                        name='att_W', dtype=tf.float32,
                        shape=[self.paras['unit'], self.paras['unit']],
                        initializer=tf.glorot_uniform_initializer()
                    )
                    self.av_b = tf.get_variable(
                        name='att_h', dtype=tf.float32,
                        shape=[self.paras['unit']],
                        initializer=tf.zeros_initializer()
                    )
                    self.av_u = tf.get_variable(
                        name='att_u', dtype=tf.float32,
                        shape=[self.paras['unit']],
                        initializer=tf.glorot_uniform_initializer()
                    )

                    self.a_laten = tf.tanh(tf.tensordot(self.outputs, self.av_W, axes=1) + self.av_b)
                    self.a_scores = tf.tensordot(self.a_laten, self.av_u, axes=1, name='scores')
                    self.a_alphas = tf.nn.softmax(self.a_scores, name='alphas')

                    self.a_con = tf.reduce_sum(self.outputs * tf.expand_dims(self.a_alphas, -1), 1)
                    self.fea_con = tf.concat( [self.outputs[:, -1, :], self.a_con], axis=1)
                    print('adversarial scope')
                    # training loss
                    self.pred = self.adv_part(self.fea_con)
                    if self.hinge:
                        self.loss = tf.losses.hinge_loss(self.gt_var, self.pred)
                    else:
                        self.loss = tf.losses.log_loss(self.gt_var, self.pred)

                    self.adv_loss = self.loss * 0

                    # adversarial loss
                    if self.adv_train:
                        print('gradient noise')
                        self.delta_adv = tf.gradients(self.loss, [self.fea_con])[0]
                        tf.stop_gradient(self.delta_adv)
                        self.delta_adv = tf.nn.l2_normalize(self.delta_adv, axis=1)
                        self.adv_pv_var = self.fea_con + self.paras['eps'] * self.delta_adv

                        scope.reuse_variables()
                        self.adv_pred = self.adv_part(self.adv_pv_var)
                        if self.hinge:
                            self.adv_loss = tf.losses.hinge_loss(self.gt_var, self.adv_pred)
                        else:
                            self.adv_loss = tf.losses.log_loss(self.gt_var, self.adv_pred)
            else:
                with tf.variable_scope('lstm_att') as scope:
                    print('adversarial scope')
                    # training loss
                    self.pred = self.adv_part(self.outputs[:, -1, :])
                    if self.hinge:
                        self.loss = tf.losses.hinge_loss(self.gt_var, self.pred)
                    else:
                        self.loss = tf.losses.log_loss(self.gt_var, self.pred)

                    self.adv_loss = self.loss * 0

                    # adversarial loss
                    if self.adv_train:
                        print('gradient noise')
                        self.delta_adv = tf.gradients(self.loss, [self.outputs[:, -1, :]])[0]
                        tf.stop_gradient(self.delta_adv)
                        self.delta_adv = tf.nn.l2_normalize(self.delta_adv, axis=1)
                        self.adv_pv_var = self.outputs[:, -1, :] + self.paras['eps'] * self.delta_adv

                        scope.reuse_variables()
                        self.adv_pred = self.adv_part(self.adv_pv_var)
                        if self.hinge:
                            self.adv_loss = tf.losses.hinge_loss(self.gt_var, self.adv_pred)
                        else:
                            self.adv_loss = tf.losses.log_loss(self.gt_var, self.adv_pred)

            # regularizer
            self.tra_vars = tf.trainable_variables('lstm_att/pre_fc')
            for var in self.tra_vars:
                self.l2_norm += tf.nn.l2_loss(var)

            self.obj_func = self.loss + self.paras['alp'] * self.l2_norm + self.paras['bet'] * self.adv_loss

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.paras['lr']).minimize(self.obj_func)


    def train(self, tune_para=False):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        best_valid_pred = np.zeros(self.val_gt.shape, dtype=float)
        best_test_pred = np.zeros(self.tes_gt.shape, dtype=float)

        best_valid_perf = {
            'Accuracy': 0, 'MCC': -2
        }
        best_test_perf = {
            'Accuracy': 0, 'MCC': -2
        }

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1
        for i in range(self.epochs):
            t1 = time()
            # first_batch = True
            tra_loss = 0.0
            tra_obj = 0.0
            l2 = 0.0
            tra_adv = 0.0
            for j in range(bat_count):
                pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
                feed_dict = {
                    self.pv_var: pv_b,
                    self.wd_var: wd_b,
                    self.gt_var: gt_b
                }
                cur_pre, cur_obj, cur_loss, cur_l2, cur_al, batch_out = sess.run(
                    (self.pred, self.obj_func, self.loss, self.l2_norm, self.adv_loss, self.optimizer),
                    feed_dict
                )

                tra_loss += cur_loss
                tra_obj += cur_obj
                l2 += cur_l2
                tra_adv += cur_al

            print('----->>>>> Training:', tra_obj / bat_count, tra_loss / bat_count, l2 / bat_count, tra_adv / bat_count)

            if not tune_para:
                tra_loss = 0.0
                tra_obj = 0.0
                l2 = 0.0
                tra_acc = 0.0
                for j in range(bat_count):
                    pv_b, wd_b, gt_b = self.get_batch(
                        j * self.batch_size)
                    feed_dict = {
                        self.pv_var: pv_b,
                        self.wd_var: wd_b,
                        self.gt_var: gt_b
                    }
                    cur_obj, cur_loss, cur_l2, cur_pre = sess.run(
                        (self.obj_func, self.loss, self.l2_norm, self.pred),
                        feed_dict
                    )
                    cur_tra_perf = evaluate(cur_pre, gt_b, self.hinge)
                    tra_loss += cur_loss
                    l2 += cur_l2
                    tra_obj += cur_obj
                    tra_acc += cur_tra_perf['Accuracy']

                print('Training:', tra_obj / bat_count, tra_loss / bat_count, l2 / bat_count, '\tTrain per:', tra_acc / bat_count)

            # test on validation set
            feed_dict = {
                self.pv_var: self.val_pv,
                self.wd_var: self.val_wd,
                self.gt_var: self.val_gt
            }
            val_loss, val_pre = sess.run(
                (self.loss, self.pred), feed_dict
            )
            cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
            print('\tVal per:', cur_valid_perf, '\tVal loss:', val_loss)

            # test on testing set
            feed_dict = {
                self.pv_var: self.tes_pv,
                self.wd_var: self.tes_wd,
                self.gt_var: self.tes_gt
            }
            test_loss, tes_pre = sess.run(
                (self.loss, self.pred), feed_dict
            )
            cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
            print('\tTest per:', cur_test_perf, '\tTest loss:', test_loss)

            if cur_valid_perf['Accuracy'] > best_valid_perf['Accuracy']:
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_pred = copy.copy(val_pre)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_pred = copy.copy(tes_pre)
                if not tune_para:
                    saver.save(sess, self.model_save_path)

                if self.hinge:
                    pred = (np.sign(tes_pre) + 1) / 2
                    for ind, p in enumerate(pred):
                        v = p[0]
                        if abs(p[0] - 0.5) < 1e-8 or np.isnan(p[0]):
                            pred[ind][0] = 0
                else:
                    pred = np.round(tes_pre)

                results = {
                    'metrics': best_test_perf,
                    'preds': tes_pre,
                    'labels': self.tes_gt,
                    'pred_date': self.test_dates,
                    'last_date': self.test_last_dates,
                    'tickers': self.test_tickers
                }

                # Save results
                with open(self.metrics_path + f'/val_metrics_sl{self.paras["seq"]}_pl{self.pred_len}.json', 'w') as f:
                    json.dump(best_valid_perf, f, indent=4)
                
                with open(self.metrics_path + f'/metrics_sl{self.paras["seq"]}_pl{self.pred_len}.json', 'w') as f:
                    json.dump(best_test_perf, f, indent=4)

                with open(self.metrics_path + f'/results_sl{self.paras["seq"]}_pl{self.pred_len}.pkl', 'wb') as f:
                    pickle.dump(results, f)

            self.tra_pv, self.tra_wd, self.tra_gt = shuffle(
                self.tra_pv, self.tra_wd, self.tra_gt, random_state=0
            )
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))

        print('\nBest Valid performance:', best_valid_perf)
        print('\tBest Test performance:', best_test_perf)
        sess.close()
        tf.reset_default_graph()
        if tune_para:
            return best_valid_perf, best_test_perf
        return best_valid_pred, best_test_pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', help='path of pv data', type=str, default='./data')
    parser.add_argument('--universe', help='universe of stocks', type=str, default='sp500')
    parser.add_argument('-l', '--seq', help='length of history', type=int, default=15)
    parser.add_argument('-n', '--pred_len', help='length of prediction', type=int, default=5)
    parser.add_argument('-u', '--unit', help='number of hidden units in lstm', type=int, default=16)
    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-3, help='alpha for l2 regularizer')
    parser.add_argument('-la', '--beta_adv', type=float, default=5e-2, help='beta for adverarial loss')
    parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-3, help='epsilon to control the scale of noise')
    parser.add_argument('-s', '--step', help='steps to make prediction', type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=1024)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
    parser.add_argument('-r', '--learning_rate', help='learning rate', type=float, default=1e-2)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')

    parser.add_argument('-qs', '--model_save_path', type=str, help='path to save model', default='./tmp/model_sp500_advALSTM')
    parser.add_argument('-m', '--model', type=str, default='Adv-ALSTM', help='Name of the model')
    parser.add_argument('-f', '--seed', type=int, default=42, help='use fixed initialization')
    parser.add_argument('-a', '--att', type=int, default=1, help='use attention model')
    parser.add_argument('-w', '--week', type=int, default=0, help='use week day data')
    parser.add_argument('-v', '--adv', type=int, default=1, help='adversarial training')
    parser.add_argument('-hi', '--hinge_lose', type=int, default=1, help='use hinge lose')
    parser.add_argument('-rl', '--reload', type=int, default=0, help='use pre-trained parameters')

    parser.add_argument('--start_date', type=str, default='2015-01-01', help='start date of the data')
    parser.add_argument('--end_train_date', type=str, default='2018-12-31', help='start date of the data')
    parser.add_argument('--end_valid_date', type=str, default='2019-12-31', help='end date of the data')
    parser.add_argument('--end_date', type=str, default='2020-12-31', help='end date of the data')

    args = parser.parse_args()

    args.model_path = f"./saved_model/sp500_alstm_seqlen{args.seq}/exp"
    metrics_path = f'./results/{args.universe}/{args.model}/{args.seed}/y{args.end_date.split("-")[0]}'

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    print(args)

    parameters = {
        'seq': int(args.seq),
        'unit': int(args.unit),
        'alp': float(args.alpha_l2),
        'bet': float(args.beta_adv),
        'eps': float(args.epsilon_adv),
        'lr': float(args.learning_rate)
    }

    model = AWLSTM(
        data_path=args.data_path,
        universe=args.universe,
        model_path=args.model_path,
        model_save_path=args.model_save_path,
        parameters=parameters,
        steps=args.step,
        epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu, start_date=args.start_date,
        tra_date=args.end_train_date, val_date=args.end_valid_date, tes_date=args.end_date, att=args.att,
        hinge=args.hinge_lose, fix_init=args.seed, adv=args.adv,
        reload=args.reload, pred_len=args.pred_len, metrics_path=metrics_path
    )

    model.train()