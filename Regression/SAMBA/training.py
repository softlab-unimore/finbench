import copy
import logging
import math
import os
import random
import time

import numpy as np
import torch


def get_logger(root, name=None, debug=True):
    #when debug is true, show DEBUG and INFO in screen
    #when debug is false, show DEBUG in file and info in both screen&file
    #INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    #critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # define the formate
        formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
        # create another handler for output log to console
        console_handler = logging.StreamHandler()
        if debug:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
            # create a handler for write log to file
            logfile = os.path.join(root, 'run.log')
            print('Creat Log File in: ', logfile)
            file_handler = logging.FileHandler(logfile, mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # add Handler to logger
        logger.addHandler(console_handler)
        if not debug:
            logger.addHandler(file_handler)
    return logger

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_device(opt):
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt

def init_optim(model, opt):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),lr=opt.lr_init)

def init_lr_scheduler(optim, opt):
    '''
    Initialize the learning rate scheduler
    '''
    #return torch.optim.lr_scheduler.StepLR(optimizer=optim,gamma=opt.lr_scheduler_rate,step_size=opt.lr_scheduler_step)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=opt.lr_decay_steps,
                                                gamma = opt.lr_scheduler_rate)

def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')

def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    return allocated_memory, cached_memory
    #print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def RRSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))

def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def PNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    indicator = torch.gt(pred - true, 0).float()
    return indicator.mean()

def oPNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    bias = (true+pred) / (2*true)
    return bias.mean()

def MARE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))

def SMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred)/(torch.abs(true)+torch.abs(pred)))


def All_Metrics(pred, true, mask1, mask2):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    #if type(pred) == np.ndarray:
    #    mae  = MAE_np(pred, true, mask1)
    #    rmse = RMSE_np(pred, true, mask1)
    #    mape = MAPE_np(pred, true, mask2)
    #    rrse = RRSE_np(pred, true, mask1)

        #corr = CORR_np(pred, true, mask1)
        #pnbi = PNBI_np(pred, true, mask1)
        #opnbi = oPNBI_np(pred, true, mask2)
    if type(pred) == torch.Tensor:
        mae  = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        rrse = RRSE_torch(pred, true, mask1)

        #pnbi = PNBI_torch(pred, true, mask1)
        #opnbi = oPNBI_torch(pred, true, mask2)
    else:
        raise TypeError
    return mae, rmse, rrse

def SIGIR_Metrics(pred, true, mask1, mask2):
    rrse = RRSE_torch(pred, true, mask1)
    # corr = CORR_torch(pred, true, 0)
    # return rrse, corr
    return rrse

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ""
    file_name = os.path.join(model_dir, epoch + "_stemgnn.pt")
    with open(file_name, "wb") as f:
        torch.save(model, f)

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)

        if val_loader != None:
            self.val_per_epoch = len(val_loader)

        self.best_path = os.path.join(self.args.get('log_dir'), 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.get('log_dir'), 'loss.png')

        if os.path.isdir(args.get('log_dir')) == False and not args.get('debug'):
            os.makedirs(args.get('log_dir'), exist_ok=True)
        self.logger = get_logger(args.get('log_dir'), name=args.get('model'), debug=args.get('debug'))
        self.logger.info('Experiment log path in: {}'.format(args.get('log_dir')))

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data.to(self.args.get('device'))
                label = target  # (..., 1)
                output = self.model(data)
                loss = self.loss(output.cpu(), label.unsqueeze(-1))

                if not torch.isnan(loss):
                    total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loss_values=[]
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.args.get('device'))
            label = target  # (..., 1)
            self.optimizer.zero_grad()

            # data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data)
            loss = self.loss(output.cpu(), label.unsqueeze(-1))
            loss.backward()

            # add max grad clipping
            if self.args.get('grad_norm'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.get('max_grad_norm'))

            self.optimizer.step()
            total_loss += loss.item()
            loss_values.append(loss.item())

            if batch_idx % self.args.get('log_step') == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))

        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(
            '**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

        # learning rate decay
        if self.args.get('lr_decay'):
            self.lr_scheduler.step()

        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.get('epochs') + 1):
            train_epoch_loss = self.train_epoch(epoch)

            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader

            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            # early stop
            if self.args.get('early_stop'):
                if not_improved_count == self.args.get('early_stop_patience'):
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.get('early_stop_patience')))
                    break

            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        if not self.args.get('debug'):
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        self.model.load_state_dict(best_model)
        # self.val_epoch(self.args.epochs, self.test_loader)
        # y1, y2 = self.test(self.model, self.args, self.test_loader, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    def test(self, data_loader, path=None):
        self.model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.args.get('device'))
                label = target
                output = self.model(data)

                y_true.append(label)
                y_pred.append(output.cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0).squeeze(-1)

        return y_pred, y_true

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
        
