import copy
import numpy as np
from tqdm import tqdm
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import utils
import torch.optim as optim
import torch.distributions as ds

from utils.load_data import load_stock_features, load_inci_matrix
from utils.args import Args
from utils.metric import evaluate_svat
from utils.optimize import Adjust_LR, init_model
from models.svat import SVAT, AdvSampler, AdvGenerator
from experiments.exp_basic import Exp_Basic

def tensor_clean(data, r_num):
    data = torch.where(torch.isnan(data), torch.full_like(data, r_num), data)
    data = torch.where(torch.isinf(data), torch.full_like(data, r_num), data)

    return data

class Exp_SVAT(Exp_Basic):
    def __init__(self, param_dict, gpus=0):
        """
        Experiment class for the Adversial HGAT model

        :params:
            - param_dict: dict      hyperparameters defined by user
            - gpus: int or list     GPU number             
        """
        super(Exp_SVAT, self).__init__(param_dict, gpus)
        args = Args(self.params[0])

        self.stock_fea, self.masks, self.gt_rr, self.base_data, self.valid_index, self.test_index, self.dates, self.tickers = load_stock_features(args)
        self.rel_mat = load_inci_matrix(self.tickers, args)

        self.total_stock_num = len(self.masks)
        self.trade_dates = self.masks.shape[1]

    def _build_model(self, args):
        """
        Build the model for experiments

        :params:
            - args: utils.args.Args   Args object storing hyperparameters
        """
        model = SVAT(
            len(self.tickers),
            args.fea_dim,
            args.hid_size,
            args.drop_rate,
            args.history_len
        )
        z_prior_sampler = AdvSampler(
            args.hid_size,
            args.z_dim,
            args.adv_hid_size
        )
        z_post_sampler = AdvSampler(
            args.hid_size * 2,
            args.z_dim,
            args.adv_hid_size
        )
        adv_generator = AdvGenerator(
            args.hid_size + args.z_dim,
            args.hid_size
        )

        model, self.device = init_model(model, self.gpus)
        z_prior_sampler, _ = init_model(z_prior_sampler, self.gpus)
        z_post_sampler, _ = init_model(z_post_sampler, self.gpus)
        adv_generator, _ = init_model(adv_generator, self.gpus)

        return model, z_prior_sampler, z_post_sampler, adv_generator

    def _get_optimizer(self, model, lr, op_type='adam'):
        if op_type == 'adam':
            model_optim = optim.Adam(model.parameters(), lr=lr)
        elif op_type == 'sgd':
            model_optim = optim.SGD(model.parameters(), lr=lr)
        elif op_type == 'momentum':
            model_optim = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif op_type == 'rmsprop':
            model_optim = optim.RMSprop(model.parameters(), lr=lr)
        
        return model_optim
    
    def _get_rank_loss(self, args, pred, ground_truth, base_data, stock_num, is_adv=False):
        if 'CASE' in args.universe:
            #pred_rr = pred
            reg_loss = nn.MSELoss(reduction='none')(pred, ground_truth)
        else:
            #pred_rr = torch.div((pred - base_data), base_data)
            reg_loss = nn.MSELoss(reduction='none')(pred, ground_truth)
        
        all_ones = torch.ones(stock_num,1).to(self.device)
        pre_pw_dif = (
            torch.matmul(pred, torch.transpose(all_ones, 0, 1)) - 
            torch.matmul(all_ones, torch.transpose(pred, 0, 1))
        )
        gt_pw_dif = (
            torch.matmul(ground_truth, torch.transpose(all_ones, 0, 1)) -
            torch.matmul(all_ones, torch.transpose(ground_truth, 0, 1))
        )
        if not is_adv:
            reg_loss = torch.sum(reg_loss)
            rank_loss = torch.sum(F.relu(-1. * pre_pw_dif * gt_pw_dif))
            total_loss = args.reg_alpha * reg_loss + rank_loss
        else:
            mask_pw = torch.sign(ground_truth)
            adv_loss = args.reg_alpha * reg_loss + torch.sum(F.relu(-1. * pre_pw_dif * gt_pw_dif), dim=1).unsqueeze(1)
            total_loss = torch.sum((mask_pw * adv_loss + (1. - mask_pw) * F.relu(args.eta - adv_loss)) * torch.abs(ground_truth))
        
        return total_loss, pred

    def _get_kl_loss(self, prior_mu, prior_std, post_mu, post_std):
        prior_pdf = ds.normal.Normal(prior_mu, prior_std)
        post_pdf = ds.normal.Normal(post_mu, post_std)
        kl_loss = ds.kl.kl_divergence(post_pdf, prior_pdf)

        return kl_loss.mean()

    def _get_batch(self, args, offset, select_all=False):
        mask_batch = self.masks[:, offset: offset + args.history_len + 1]
        mask_batch = np.min(mask_batch, axis=1)
        if select_all:
            stock_idx = np.arange(0, self.total_stock_num, dtype=np.int64)
        else:
            stock_idx = np.argwhere(mask_batch > 1e-8)
            stock_idx = stock_idx.reshape(len(stock_idx))

        if 'CASE' in args.universe:
            base_data = np.expand_dims(self.base_data[stock_idx, offset+args.history_len], axis=1)
        else:
            base_data = np.expand_dims(self.base_data[stock_idx, offset+args.history_len-1], axis=1)
        return self.stock_fea[stock_idx, offset:offset+args.history_len], \
               self.stock_fea[stock_idx, offset:offset+args.history_len+1, -1], \
               self.rel_mat[stock_idx, :], \
               np.expand_dims(self.gt_rr[stock_idx, offset+args.history_len+args.pred_len-1], axis=1), \
               base_data, \
               stock_idx, \
               np.expand_dims(mask_batch, axis=1)

    def valid(self, start_idx, end_idx, args):
        with torch.no_grad():
            self.model.eval()
            self.z_prior_sampler.eval()
            self.z_post_sampler.eval()
            self.adv_generator.eval()
            cur_valid_pred = np.zeros([self.total_stock_num, end_idx-start_idx], dtype=float)
            cur_valid_gt = np.zeros([self.total_stock_num, end_idx-start_idx], dtype=float)
            cur_valid_mask = np.zeros([self.total_stock_num, end_idx-start_idx], dtype=float)
            val_rank_loss = 0.0
            # rel_sparse = sparse.coo_matrix(self.rel_mat)
            # incidence_edge = utils.from_scipy_sparse_matrix(rel_sparse)
            # hyp_input = incidence_edge[0].to(self.device)

            for cur_offset in tqdm(range(start_idx, end_idx)):
                fea_batch, _, rel_mat, gt_batch, base_data, _, mask_batch = self._get_batch(args, cur_offset, True)

                rel_sum = np.sum(rel_mat, axis=0)
                rel_id = np.argwhere(rel_sum > 1.)[:, 0]
                rel_mat = rel_mat[:, rel_id]
                rel_sparse = sparse.coo_matrix(rel_mat)
                incidence_edge = utils.from_scipy_sparse_matrix(rel_sparse)
                hyp_input = incidence_edge[0].to(self.device)

                att_fea, hg_fea, output = self.model(torch.FloatTensor(fea_batch).to(self.device), hyp_input)
                cur_rank_loss, curr_rank_score = self._get_rank_loss(
                    args,
                    output.reshape((self.total_stock_num,1)), 
                    torch.FloatTensor(gt_batch).to(self.device),
                    torch.FloatTensor(base_data).to(self.device),
                    self.total_stock_num,
                    False
                )
                val_rank_loss += cur_rank_loss.detach().cpu().item()
                cur_valid_gt[:, cur_offset-start_idx] = copy.copy(gt_batch[:, 0])
                cur_valid_mask[:, cur_offset-start_idx] = copy.copy(mask_batch[:, 0])
                curr_rank_score = curr_rank_score.detach().cpu().numpy().reshape((self.total_stock_num,1))
                cur_valid_pred[:, cur_offset-start_idx] = copy.copy(curr_rank_score[:, 0])
            cur_valid_perf = evaluate_svat(cur_valid_pred, cur_valid_gt, cur_valid_mask)
            val_rank_loss = val_rank_loss / (self.test_index - self.valid_index)

            return val_rank_loss, cur_valid_perf, cur_valid_pred, cur_valid_gt, cur_valid_mask

    def train(self):
        best_perf = {'sharpe5': -np.inf}

        best_preds = None
        best_gt = None
        mask = None

        best_valid_pred = None
        best_valid_labels = None
        valid_mask = None

        for k, param in enumerate(self.params):
            args = Args(param)

            print('============== start training ' + str(k) + '-th parameters ==============')

            self.model, self.z_prior_sampler, self.z_post_sampler, self.adv_generator = self._build_model(args)
            model_optim = self._get_optimizer(self.model, args.learning_rate, 'adam')
            z_prior_optim = self._get_optimizer(self.z_prior_sampler, args.adv_lr, 'adam')
            z_post_optim = self._get_optimizer(self.z_post_sampler, args.adv_lr, 'adam')
            adv_gen_optim = self._get_optimizer(self.adv_generator, args.adv_lr, 'adam')
            lr_adjuster = Adjust_LR(patience=args.patience)
            batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)

            for i in range(args.epochs):
                self.model.train()
                self.z_prior_sampler.train()
                self.z_post_sampler.train()
                self.adv_generator.train()
                np.random.shuffle(batch_offsets)
                tra_rank_loss = 0.0

                for j in tqdm(range(self.valid_index), ncols=85):
                # for j in tqdm(range(3), ncols=85):
                    fea_batch, _, rel_mat, gt_batch, base_data, stock_idx, _ = self._get_batch(args, batch_offsets[j])
                    rel_sum = np.sum(rel_mat, axis=0)
                    rel_id = np.argwhere(rel_sum > 1.)[:,0]
                    rel_mat = rel_mat[:, rel_id]
                    rel_sparse = sparse.coo_matrix(rel_mat)
                    incidence_edge = utils.from_scipy_sparse_matrix(rel_sparse)
                    hyp_input = incidence_edge[0].to(self.device)
                    batch_size = len(fea_batch)

                    self.model.eval()
                    z_prior_optim.zero_grad()
                    z_post_optim.zero_grad()
                    adv_gen_optim.zero_grad()
                    att_fea, hg_fea, output = self.model(torch.FloatTensor(fea_batch).to(self.device), hyp_input, stock_idx)
                    fea_con = att_fea if args.adv == 'Attention' else hg_fea
                    loss_for_adv, _ = self._get_rank_loss(
                        args,
                        output.reshape((batch_size,1)), 
                        torch.FloatTensor(gt_batch).to(self.device),
                        torch.FloatTensor(base_data).to(self.device),
                        batch_size,
                        False
                    )
                    z_prior_mu, z_prior_std = self.z_prior_sampler(fea_con.detach())
                    z_prior_mu = tensor_clean(z_prior_mu, 0.)
                    z_prior_std = tensor_clean(z_prior_std, 1.)
                    grad = torch.autograd.grad(loss_for_adv, [fea_con])[0].detach()
                    norm_grad = args.adv_eps * nn.functional.normalize(grad, p=2, dim=1)
                    z_post_mu, z_post_std = self.z_post_sampler(torch.cat([fea_con.detach(), norm_grad], 1))
                    z_post_mu = tensor_clean(z_post_mu, 0.)
                    z_post_std = tensor_clean(z_post_std, 1.)
                    z_eps = torch.randn_like(z_post_std)
                    z_post = z_post_mu + z_post_std * z_eps
                    delta_g = self.adv_generator(torch.cat([fea_con.detach(), z_post], 1))
                    delta = args.adv_eps * nn.functional.normalize(delta_g, p=2, dim=1)

                    self.model.train()
                    model_optim.zero_grad()
                    att_fea, hg_fea, output = self.model(torch.FloatTensor(fea_batch).to(self.device), hyp_input, stock_idx)
                    fea_con = att_fea if args.adv == 'Attention' else hg_fea
                    e = hyp_input if args.adv == 'Attention' else None
                    x_adv = fea_con + delta
                    adv_output = self.model.adv(x_adv, e)

                    origin_loss, _ = self._get_rank_loss(
                        args,
                        output.reshape((batch_size,1)), 
                        torch.FloatTensor(gt_batch).to(self.device),
                        torch.FloatTensor(base_data).to(self.device),
                        batch_size,
                        False
                    )
                    tra_rank_loss += origin_loss.item()
                    adv_loss, _ = self._get_rank_loss(
                        args,
                        adv_output.reshape((batch_size,1)), 
                        torch.FloatTensor(gt_batch).to(self.device),
                        torch.FloatTensor(base_data).to(self.device),
                        batch_size,
                        True
                    )
                    kl_loss = self._get_kl_loss(z_prior_mu, z_prior_std, z_post_mu, z_post_std)
                    if torch.isnan(adv_loss).any() or torch.isnan(origin_loss).any():
                        print("Sample: {}, Origin Loss: {}, Adv Loss: {}".format(j, origin_loss, adv_loss))
                    total_loss = origin_loss + adv_loss + args.kl_lambda * kl_loss

                    total_loss.backward()
                    model_optim.step()
                    z_prior_optim.step()
                    z_post_optim.step()
                    adv_gen_optim.step()

                tra_rank_loss /= (self.valid_index)
                # tra_rank_loss /= 3

                print('Train Rank Loss: {}, Total Loss: {}'.format(tra_rank_loss, total_loss))
                lr_adjuster(model_optim, tra_rank_loss, i+1, args)

                cur_valid_loss, cur_valid_perf, cur_valid_pred, cur_valid_labels, valid_mask = self.valid(self.valid_index, self.test_index, args)
                if cur_valid_perf['btl5'] < 0. and cur_valid_perf['sharpe5'] > best_perf['sharpe5']:
                    best_perf = cur_valid_perf.copy()
                    self.best_args = args

                    best_valid_pred = cur_valid_pred.copy()
                    best_valid_labels = cur_valid_labels.copy()

                    end_idx = self.trade_dates - args.history_len - args.pred_len + 1
                    cur_test_loss, cur_test_perf, best_preds, best_gt, mask = self.valid(self.test_index, end_idx, args)

                print('Valid Rank Loss: {}'.format(cur_valid_loss))
                print('====> Valid preformance: btl5: {}, sharpe5: {}'.format(cur_valid_perf['btl5'], cur_valid_perf['sharpe5']))

        print('============== The Best parameters ==============')
        print(self.best_args)
        print('====> Best valid preformance:')
        print('{}'.format(best_perf))

        return best_preds, best_gt, mask, best_valid_pred, best_valid_labels, valid_mask
