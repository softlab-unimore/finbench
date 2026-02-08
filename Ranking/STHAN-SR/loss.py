import torch
import torch.nn.functional as F


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


def trr_loss_mse_rank(pred, base_price, ground_truth, mask, alpha, no_stocks, device='cuda'):
    return_ratio = torch.div((pred- base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks,1).to(device)
    pre_pw_dif =  (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1))
                    - torch.matmul(all_ones, torch.transpose(return_ratio, 0, 1)))
    gt_pw_dif = (
            torch.matmul(all_ones, torch.transpose(ground_truth,0,1)) -
            torch.matmul(ground_truth, torch.transpose(all_ones, 0,1))
        )

    mask_pw = torch.matmul(mask, torch.transpose(mask, 0,1))
    rank_loss = torch.mean(
            F.relu(((pre_pw_dif*gt_pw_dif)*mask_pw)))
    loss = reg_loss + alpha*rank_loss
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio