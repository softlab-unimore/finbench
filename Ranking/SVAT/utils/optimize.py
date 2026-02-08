import numpy as np
import torch
import torch.nn as nn
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def init_model(model, gpus=None):
    if gpus is not None:
        if type(gpus) == list:
            model = nn.DataParallel(model, device_ids=gpus)
            device = torch.device('cuda:{}'.format(gpus[0]))
        else:
            device = torch.device('cuda:{}'.format(gpus))
            model = model.to(device)
    else:
        device = torch.device('cpu')
        model = model.to(device)
    
    return model, device
    

class Adjust_LR:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.adjust_times = 0
        self.loss_min = np.inf
    
    def __call__(self, optimizer, cur_loss, epoch, args):
        if cur_loss < self.loss_min:
            self.loss_min = cur_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.adjust_times += 1
                lr = args.learning_rate * (0.1**self.adjust_times)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                print('Updating learning rate to {}'.format(lr))
                self.counter = 0