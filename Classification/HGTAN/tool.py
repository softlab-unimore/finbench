import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import time
import torch.utils.data as Data
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef

from load_data import get_batch


def cal_performance(pred, gold, smoothing=False):

    loss = cal_loss(pred, gold, smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)

    n_correct = pred.eq(gold)
    n_correct = n_correct.sum().item()

    return loss, n_correct

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = 3

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')
    return loss


def get_metrics(preds, labels):
    preds = np.argmax(np.array(preds), axis=2)
    preds = preds.reshape(-1)
    labels = np.array(labels).reshape(-1)

    metrics = {
        'F1': f1_score(labels, preds, average='macro'),
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds, average='macro'),
        'Recall': recall_score(labels, preds, average='macro'),
        'MCC': matthews_corrcoef(labels, preds)
    }
    return metrics


def eval_model(model, test_data, adj, H, device, args):

    log_dir = args.save_model + '.chkpt'
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        # args.load_state_dict(checkpoint['settings'])
        start_epoch = checkpoint['epoch']
        print('load epoch {} successfully！'.format(start_epoch))
    else:
        print('no save model, start training from the beginning！')

    start = time.time()
    test_loss, test_accu, preds, labels = eval_epoch(model, test_data, adj, H, device, args)
    print(' - (test) loss:{loss:8.5f}, accuracy:{accu:3.3f}%, elapse: {elapse:3.3f} min'.format(
            loss=test_loss, accu=100 * test_accu, elapse=(time.time() - start) / 60))

    return preds, labels


def train_epoch(model,training_data, optimizer, device, smoothing, args, adj, H):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    total_accu = 0
    n_count = 0

    for step, (eod, gt) in enumerate(training_data):

        Eod, Gt, H_,adj_= eod.to(device), gt.to(device), H.to(device), adj.to(device)

        # forward
        optimizer.zero_grad()
        pred = model(Eod,H_,adj_,args.hidden)

        # backward
        loss, n_correct = cal_performance(pred, Gt, smoothing=smoothing)
        loss.backward()

        optimizer.step_and_update_lr()

        total_loss += loss.item()
        total_accu += n_correct
        n_count += Eod.size(0) * Eod.size(1)

    epoch_loss = total_loss / n_count
    accuracy = total_accu / n_count
    return epoch_loss, accuracy


def eval_epoch(model, validation_data, adj, H, device,args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total_accu = 0
    n_count = 0
    valid_pred = []
    all_gt = []
    with torch.no_grad():
        for step, (eod, gt) in enumerate(validation_data):

            # prepare data
            Eod, Gt, H_, adj_,= eod.to(device), gt.to(device), H.to(device), adj.to(device)

            # forward
            pred = model(Eod, H_, adj_, args.hidden)
            loss, n_correct = cal_performance(pred, Gt, smoothing=False)
            # pred = pred.max(1)[1].view_as(Gt)
            pred = pred.view(Gt.shape[0], Gt.shape[1], 3)
            pred = pred.cuda().data.cpu().numpy()
            valid_pred.append(pred)
            all_gt.append(Gt.cpu().numpy())

            total_loss += loss.item()
            total_accu += n_correct
            n_count += Eod.size(0) * Eod.size(1)

    epoch_loss = total_loss / n_count
    accuracy = total_accu / n_count

    valid_pred = [row for arr in valid_pred for row in arr]
    all_gt = [row[:, None] for arr in all_gt for row in arr]

    return epoch_loss, accuracy, valid_pred, all_gt


def train(model, training_data, validation_data, adj, H, optimizer, device, args, metrics_path):
    ''' Start training '''

    valid_accus = []
    Train_Loss_list = []
    Train_Accuracy_list = []
    Val_Loss_list = []
    Val_Accuracy_list = []

    for epoch_i in range(args.epochs):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=args.label_smoothing, args=args, adj=adj, H=H)

        Train_Loss_list.append(train_loss)
        Train_Accuracy_list.append(100 * train_accu)

        print(' - (Training) loss:{loss:8.5f}, accuracy:{accu:3.3f}%, elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_accu, val_preds, val_labels = eval_epoch(model, validation_data, adj, H, device, args=args)
        Val_Loss_list.append(valid_loss)
        Val_Accuracy_list.append(100 * valid_accu)

        print(' - (Validation) loss:{loss:8.5f}, accuracy:{accu:3.3f}%. elapse: {elapse:3.3f} min'.format(
            loss=valid_loss, accu=100 * valid_accu, elapse=(time.time() - start) / 60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'epoch': epoch_i
        }

        if args.save_model:
            if args.save_mode == 'all':
                model_name = args.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                model_name = args.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

                    val_metrics = get_metrics(val_preds, val_labels)
                    val_metrics = {k: float(v) for k, v in val_metrics.items()}

                    with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                        json.dump(val_metrics, f, indent=4)

    print('{Best:3.3f}\n'.format(Best=100 * max(valid_accus)))
    print('{Best_epoch: 4.0f}\n'.format(Best_epoch=valid_accus.index(max(valid_accus))))


def prepare_dataloaders(eod_data, gt_data, valid_index, test_index, args):
    # ========= Preparing DataLoader =========#
    EOD, GT = [], []

    for i in range(eod_data.shape[1] - args.seq_len - args.pred_len + 1):
        eod, gt = get_batch(eod_data, gt_data, i, args.seq_len, args.pred_len)
        EOD.append(eod)
        GT.append(gt)

    train_eod, train_gt = EOD[:valid_index], GT[:valid_index]
    valid_eod, valid_gt = EOD[valid_index:test_index], GT[valid_index:test_index]
    test_eod, test_gt = EOD[test_index:], GT[test_index:]

    train_eod, valid_eod, test_eod = np.array(train_eod), np.array(valid_eod), np.array(test_eod)
    train_eod, valid_eod, test_eod = torch.FloatTensor(train_eod), torch.FloatTensor(valid_eod), torch.FloatTensor(test_eod)
    train_gt, valid_gt, test_gt = torch.LongTensor(train_gt), torch.LongTensor(valid_gt), torch.LongTensor(test_gt)

    train_dataset = Data.TensorDataset(train_eod, train_gt)
    valid_dataset = Data.TensorDataset(valid_eod, valid_gt)
    test_dataset = Data.TensorDataset(test_eod, test_gt)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    return train_loader, valid_loader, test_loader

