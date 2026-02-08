import math
import copy
import heapq

from scipy.stats import spearmanr
from tqdm import tqdm
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score
from empyrical.stats import max_drawdown, downside_risk, calmar_ratio

def evaluate(prediction, ground_truth, mask, topk=5):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask) ** 2. / np.sum(mask)
    bt_long5 = 1.0
    sharpe_li5 = []
    ndcg_score_top5 = []

    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top5 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top5) < topk:
                gt_top5.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])
        pre_top5 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top5) < topk:
                pre_top5.add(cur_rank)
        if topk == 1:
            ndcg_score_top5.append(0.)
        else:
            ndcg_score_top5.append(ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1)))
        
        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= topk
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top5)
    
    performance['ndcg_score_top5'] = np.mean(np.array(ndcg_score_top5))
    performance['btl5'] = bt_long5 - 1
    sharpe_li5 = np.array(sharpe_li5)
    performance['btl5_array'] = sharpe_li5
    performance['sharpe5'] = ((np.mean(sharpe_li5) - (0.018/365.0))/np.std(sharpe_li5))*15.87 #To annualize
    return performance


def evaluate_svat(prediction, ground_truth, mask, topk=5):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    bt_long5 = 1.0
    sharpe_li5 = []

    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top5 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top5) < topk:
                gt_top5.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])
        pre_top5 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top5) < topk:
                pre_top5.add(cur_rank)

        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= topk
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top5)

    performance['btl5'] = bt_long5 - 1
    sharpe_li5 = np.array(sharpe_li5)
    performance['sharpe5'] = ((np.mean(sharpe_li5) - (0.018 / 365.0)) / np.std(sharpe_li5)) * 15.87  # To annualize
    return performance


def get_metrics(pred, target):
    T = pred.shape[0]
    ICs = []
    RankICs = []

    for t in range(T):
        p = pred[t]
        r = target[t]

        ICs.append(np.corrcoef(p, r)[0, 1])
        RankICs.append(spearmanr(p, r).correlation)

    ICs = np.array(ICs)
    RankICs = np.array(RankICs)

    IC_mean = ICs.mean()
    RankIC_mean = RankICs.mean()
    ICIR = IC_mean / ICs.std(ddof=1)
    RankICIR = RankIC_mean / RankICs.std(ddof=1)

    return {'IC': IC_mean, 'RankIC': RankIC_mean, 'ICIR': ICIR, 'RankICIR': RankICIR}

