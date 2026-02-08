import copy
import random

import numpy as np
import torch
from scipy import sparse
from torch import optim
from torch_geometric import utils
from tqdm import tqdm

from evaluator import evaluate
from hgat import HGAT
import torch.nn as nn

from loss import trr_loss_mse_rank


class ReRaLSTM:
    def __init__(self, eod_data, gt_data, price_data, mask_data, tickers, inci_matrix, valid_index, test_index, seq_len,
                 steps=1, epochs=50, flat=False, gpu=False, device='cpu', alpha=0.1, lr=0.001, early_stop=5):

        self.steps = steps
        self.seq_len = seq_len

        self.eod_data = eod_data
        self.gt_data = gt_data
        self.mask_data = mask_data
        self.price_data = price_data
        self.tickers = tickers
        self.valid_index = valid_index
        self.test_index = test_index
        self.trade_dates = self.mask_data.shape[1]

        self.epochs = epochs
        self.flat = flat
        self.batch_size = len(self.tickers)
        self.fea_dim = 5
        self.alpha = alpha
        self.lr = lr
        self.inci_matrix = inci_matrix
        self.early_stop = early_stop

        self.gpu = gpu
        self.device = device

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        mask_batch = self.mask_data[:, offset: offset + self.seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + self.seq_len, :], \
            np.expand_dims(mask_batch, axis=1), \
            np.expand_dims(self.price_data[:, offset + self.seq_len - 1], axis=1), \
            np.expand_dims(self.gt_data[:, offset + self.seq_len + self.steps - 1], axis=1)

    def train(self):
        global df
        print('device name:', self.device)

        model = HGAT(self.batch_size).to(self.device)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        optimizer_hgat = optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)

        inci_sparse = sparse.coo_matrix(self.inci_matrix)
        incidence_edge, edge_weight = utils.from_scipy_sparse_matrix(inci_sparse)

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)

        best_ndcg_score = 0
        best_loss = None

        early_stop = 0
        best_gt = None
        best_pred = None
        best_valid_gt = None
        best_valid_pred = None
        for i in range(self.epochs):
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            model.train()
            for j in tqdm(range(self.valid_index)):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(batch_offsets[j])

                optimizer_hgat.zero_grad()

                num_hyperedges = int(incidence_edge[1].max().item() + 1)
                hyperedge_attr = torch.ones((num_hyperedges, 32), device=self.device)

                output = model(torch.FloatTensor(emb_batch).to(self.device), incidence_edge.long().to(self.device),
                               hyperedge_attr.to(self.device))

                cur_loss, cur_reg_loss, cur_rank_loss, curr_rr_train = trr_loss_mse_rank(
                    output.reshape((self.batch_size, 1)),
                    torch.FloatTensor(price_batch).to(self.device),
                    torch.FloatTensor(gt_batch).to(self.device),
                    torch.FloatTensor(mask_batch).to(self.device),
                    self.alpha, self.batch_size
                )

                cur_loss.backward()
                optimizer_hgat.step()
                tra_loss += cur_loss.detach().cpu().item()
                tra_reg_loss += cur_reg_loss.detach().cpu().item()
                tra_rank_loss += cur_rank_loss.detach().cpu().item()

            print('Train Loss:',
                  tra_loss / (self.valid_index - self.seq_len - self.steps + 1),
                  tra_reg_loss / (self.valid_index - self.seq_len - self.steps + 1),
                  tra_rank_loss / (self.valid_index - self.seq_len - self.steps + 1))

            with torch.no_grad():
                # test on validation set
                cur_valid_pred = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_gt = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_mask = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                val_loss = 0.0
                val_reg_loss = 0.0
                val_rank_loss = 0.0
                model.eval()
                for cur_offset in range(self.valid_index, self.test_index):
                    emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)

                    num_hyperedges = int(incidence_edge[1].max().item() + 1)
                    hyperedge_attr = torch.ones((num_hyperedges, 32), device=self.device)

                    output_val = model(torch.FloatTensor(emb_batch).to(self.device), incidence_edge.to(self.device),
                                       hyperedge_attr.to(self.device))
                    
                    cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(
                        output_val,
                        torch.FloatTensor(price_batch).to(self.device),
                        torch.FloatTensor(gt_batch).to(self.device),
                        torch.FloatTensor(mask_batch).to(self.device),
                        self.alpha, self.batch_size
                    )

                    cur_rr = cur_rr.detach().cpu().numpy().reshape((self.batch_size, 1))
                    val_loss += cur_loss.detach().cpu().item()
                    val_reg_loss += cur_reg_loss.detach().cpu().item()
                    val_rank_loss += cur_rank_loss.detach().cpu().item()
                    cur_valid_pred[:, cur_offset - self.valid_index] = copy.copy(cur_rr[:, 0])
                    cur_valid_gt[:, cur_offset - self.valid_index] = copy.copy(gt_batch[:, 0])
                    cur_valid_mask[:, cur_offset - self.valid_index] = copy.copy(mask_batch[:, 0])
                
                print('Valid MSE:',
                      val_loss / (self.test_index - self.valid_index),
                      val_reg_loss / (self.test_index - self.valid_index),
                      val_rank_loss / (self.test_index - self.valid_index))
                cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
                print('\t Valid preformance:', cur_valid_perf)

                # test on testing set
                cur_test_pred = np.zeros(
                    [len(self.tickers), self.trade_dates - self.seq_len - self.steps + 1 - self.test_index],
                    dtype=float
                )
                cur_test_gt = np.zeros(
                    [len(self.tickers), self.trade_dates - self.seq_len - self.steps + 1 - self.test_index],
                    dtype=float
                )
                cur_test_mask = np.zeros(
                    [len(self.tickers), self.trade_dates - self.seq_len - self.steps + 1 - self.test_index],
                    dtype=float
                )
                test_loss = 0.0
                test_reg_loss = 0.0
                test_rank_loss = 0.0
                model.eval()
                for cur_offset in range(self.test_index, self.trade_dates - self.seq_len - self.steps + 1):
                    emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)

                    output_test = model(torch.FloatTensor(emb_batch).to(self.device), incidence_edge.to(self.device),
                                        hyperedge_attr.to(self.device))

                    cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(
                        output_test,
                        torch.FloatTensor(price_batch).to(self.device),
                        torch.FloatTensor(gt_batch).to(self.device),
                        torch.FloatTensor(mask_batch).to(self.device),
                        self.alpha, self.batch_size
                    )

                    cur_rr = cur_rr.detach().cpu().numpy().reshape((self.batch_size, 1))
                    test_loss += cur_loss.detach().cpu().item()
                    test_reg_loss += cur_reg_loss.detach().cpu().item()
                    test_rank_loss += cur_rank_loss.detach().cpu().item()
                    cur_test_pred[:, cur_offset - self.test_index] = copy.copy(cur_rr[:, 0])
                    cur_test_gt[:, cur_offset - self.test_index] = copy.copy(gt_batch[:, 0])
                    cur_test_mask[:, cur_offset - self.test_index] = copy.copy(mask_batch[:, 0])

                print('Test MSE:',
                      test_loss / (self.trade_dates - self.seq_len - self.steps + 1 - self.test_index),
                      test_reg_loss / (self.trade_dates - self.seq_len - self.steps + 1 - self.test_index),
                      test_rank_loss / (self.trade_dates - self.seq_len - self.steps + 1 - self.test_index))
                cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
                print('\t Test performance:', cur_test_perf)

                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss

                    best_valid_gt = cur_valid_gt
                    best_valid_pred = cur_valid_pred
                    best_gt = cur_test_gt
                    best_pred = cur_test_pred

                    early_stop = 0

                else:
                    early_stop += 1

                if early_stop >= self.early_stop:
                    break


        return best_gt, best_pred, best_valid_gt, best_valid_pred