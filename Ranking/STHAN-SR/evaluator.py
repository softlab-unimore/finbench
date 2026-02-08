import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


def evaluate(prediction, ground_truth, mask):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'

    performance = {}

    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top5 = set()
    
        
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])

        pre_top5 = set()

        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
        performance['ndcg_score_top5'] = ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1))

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