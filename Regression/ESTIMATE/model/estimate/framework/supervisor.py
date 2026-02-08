import json
import os
import pickle

import torch
import yaml
import numpy as np
import utils.model as utls
from utils.model import save_results

from model.estimate.framework.data_loader import DataLoader
from model.estimate.framework.model import Model

class Supervisor():
    def __init__(self, config):
        self._config = config
        self._symbols = config["data"]["symbols"]
        
        # backtest
        with open(config["backtest"]["config_path"], "r") as f:
            self._backtest_config = yaml.safe_load(f)
        self._start_backtest = self._backtest_config["start_backtest"]
        self._end_backtest = self._backtest_config["end_backtest"]

        # model
        self._hidden_dim = config["model"]["hidden_dim"]
        self._dropout = config["model"]["dropout"]
        self._batch_size = config["model"]["batch_size"]
        self._epochs = config["model"]["epochs"]
        self._lr = config["model"]["learning_rate"]
        self._cuda = config["model"]["cuda"]
        self._resume = config["model"]["resume"]
        self._confidence_threshold = config["model"]["confidence_threshold"]
        self._earlystop = config["model"]["earlystop"]
        self._save_best = config["model"]["save_best"]
        self._lr_decay = config["model"]["lr_decay"]
        self._eval_iter = config["model"]["eval_iter"]
        self._rnn_units = config["model"]["rnn_units"]
        self._verify_threshold = config["model"]["verify_threshold"]
        self._pretrained_log = config["model"]["pretrained_log"]
        self._model_name = config["model"]["model_name"]
        self._seed = config["model"]["seed"]
        
        # Get data
        self._start_train = config["data"]["start_train"]
        self._end_train = config["data"]["end_train"]
        self._start_valid = config["data"]["start_valid"]
        self._end_valid = config["data"]["end_valid"]
        self._start_test = config["data"]["start_test"]
        self._end_test = config["data"]["end_test"]
        self._n_step_ahead = config["data"]["n_step_ahead"]
        self._his_window = config["data"]["history_window"]
        self._universe = config["data"]["universe"]
        self._data_loader = DataLoader(self._config)
        self._X_train, self._y_train, self._X_valid, self._y_valid, self._X_test, self._y_test, self._hypergraphsnapshot, self._incidence_edges, self._buy_prob_threshold, self._sell_prob_threshold, self.last_dates, self.test_pred_date = self._data_loader.get_data(
            self._start_train, self._end_train, self._start_valid, self._end_valid, self._start_test, self._end_test)
        # Initialize model
        num_stocks = len(self._symbols)
        self._model = Model(snapshots=self._hypergraphsnapshot, num_stock=num_stocks, history_window=self._his_window, num_feature=self._X_train.shape[3], embedding_dim = self._hidden_dim, rnn_hidden_unit = self._rnn_units, drop_prob = self._dropout)
    
    def train(self):
        print("Begin training...")
        # Initialization

        X_train_tensor = torch.FloatTensor(self._X_train)
        y_train_tensor = torch.FloatTensor(self._y_train)
        X_valid_tensor = torch.FloatTensor(self._X_valid)
        model = self._model
        edges = self._incidence_edges
        if self._cuda:
            model = model.cuda()
            edges = edges.cuda()
            X_train_tensor = X_train_tensor.cuda()
            X_valid_tensor = X_valid_tensor.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr, weight_decay=self._lr_decay)
        epoch = 1

        # Resume training
        if self._resume:
            best_model_file = os.path.join(self._pretrained_log, "model-best.pth")
            checkpoint = torch.load(best_model_file)
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(
                checkpoint['optimizer'])
            print("Resume training at epoch {} - loss {}".format(epoch, loss))
        
        # Begin training
        model.train()
        best_acc = -999
        confidence = 0
        while epoch <= self._epochs:
            indices = np.arange(self._X_train.shape[0])
            np.random.shuffle(indices)
            X_train_epoch = X_train_tensor[indices]
            y_train_epoch = y_train_tensor[indices]
            train_acc = 0
            train_loss = 0
            for i in range(0, len(indices)-self._batch_size, self._batch_size):
                train_batch = X_train_epoch[i:i+self._batch_size]
                targets = y_train_epoch[i:i+self._batch_size]
                if self._cuda:
                    train_batch = train_batch.cuda()
                    targets = targets.cuda()
                optimizer.zero_grad()
                outputs = model(train_batch)
                loss = torch.norm(outputs - targets)
                loss.backward()        
                optimizer.step()
                train_loss += loss.cpu().item()
            with torch.no_grad():    
            
                train_preds = model(X_train_tensor).cpu().numpy()[:, :,  -1].flatten()    
                train_gt = self._y_train[:, :, -1].flatten()
                train_acc = train_preds * train_gt
                train_acc = len(train_acc[train_acc > 0]) / len(train_acc)

            valid_acc = 0
            # if epoch % self._eval_iter == 0:
            with torch.no_grad():
                valid_preds = model(X_valid_tensor).cpu().numpy()[:-self._n_step_ahead, :, -1:]
                valid_preds_flatten = valid_preds.flatten()
                valid_gt = self._y_valid[:-self._n_step_ahead, :, -1:]
                valid_gt_flatten = valid_gt.flatten()
                valid_acc = valid_preds_flatten * valid_gt_flatten
                valid_acc = len(valid_acc[valid_acc > 0]) / len(valid_acc)
                hc_threshold = np.mean(valid_gt_flatten, axis=0)
                hc_eval = utls.eval_long_acc(valid_gt_flatten, valid_preds_flatten, hc_threshold)[0]
                print("Epoch {}: train_loss {}, train_acc {:.3f}, eval_acc {:.3f}, hc_eval_acc {:.3f}".format(epoch, train_loss, train_acc, valid_acc, hc_eval))
                # save_results([epoch, train_loss, train_acc, valid_acc, hc_eval], os.path.join(self._pretrained_log, "train_log.csv"))

            if self._save_best:
                if valid_acc > best_acc and train_acc > 0.5:
                    best_acc = valid_acc
                    utls.torch_save_whole_model(model, os.path.join(self._pretrained_log, "model-best.pth"))
                    utls.dump_config(self._config, os.path.join(self._pretrained_log, "config.yaml"))

                    dict_metrics = utls.eval_all_metrics(valid_gt.squeeze(-1), valid_preds.squeeze(-1))
                    print("Results are saved in the log file!")

                    val_metrics = {
                        'MSE': dict_metrics['mse'],
                        'MAE': dict_metrics['mae'],
                        'RMSE': dict_metrics['rmse'],
                        'R2': dict_metrics['r2'],
                    }

                    metrics_path = f'./results/{self._universe}/{self._model_name}/{self._seed}/y{self._start_test.split("-")[0]}'
                    os.makedirs(metrics_path, exist_ok=True)

                    with open(f'{metrics_path}/val_metrics_sl{self._his_window}_pl{self._n_step_ahead}.json', 'w') as f:
                        json.dump(val_metrics, f, indent=4)

            if train_acc >= self._confidence_threshold:
                confidence += 1
            else:
                confidence = 0
            if confidence >= self._earlystop:
                print("Early stop due to train acc reach")
                break

            epoch += 1

        if not self._save_best:
            utls.torch_save_whole_model(model, os.path.join(self._pretrained_log, "model-best.pth"))
            utls.dump_config(self._config, os.path.join(self._pretrained_log, "config.yaml"))
        print("Done training!")

    def test(self):
        print("Begin testing...")
        if self._cuda:
            checkpoint = torch.load(os.path.join(self._pretrained_log, "model-best.pth"), weights_only=False)
        else:
            checkpoint = torch.load(os.path.join(self._pretrained_log, "model-best.pth"), map_location='cpu')

        #self._model.load_state_dict(checkpoint)
        self._model = checkpoint
        self._model.eval()
        X_test = torch.FloatTensor(self._X_test)
        model = self._model
        edges = self._incidence_edges
        if self._cuda:
            model = model.cuda()
            X_test = X_test.cuda()
            edges = edges.cuda()

        with torch.inference_mode():
            # Accuracy
            test_preds = np.zeros(shape=(len(X_test), len(self._symbols), self._his_window))
            for i in range(0, len(X_test), self._batch_size):
                test_preds[i:i+self._batch_size] = model(X_test[i:i+self._batch_size]).cpu().detach().numpy()
            remainder = len(X_test) % self._batch_size
            test_preds[-remainder:] = model(X_test[-remainder:]).cpu().detach().numpy()

            test_preds, labels = test_preds[:, :, -1], self._y_test[:, :, -1]
            dict_metrics = utls.eval_all_metrics(labels, test_preds)
            print("Results are saved in the log file!")

            preds = [test_preds[i][:, None] for i in range(len(test_preds))]
            labels = [labels[i][:, None] for i in range(len(labels))]

            metrics = {
                'MSE': dict_metrics['mse'],
                'MAE': dict_metrics['mae'],
                'RMSE': dict_metrics['rmse'],
                'R2': dict_metrics['r2'],
            }

            results = {
                'metrics': metrics,
                'preds': preds,
                'labels': labels,
                'pred_date': self.test_pred_date,
                'last_date': self.last_dates,
                'tickers': [self._symbols] * len(preds)
            }

            metrics_path = f'./results/{self._universe}/{self._model_name}/{self._seed}/y{self._start_test.split("-")[0]}'
            os.makedirs(metrics_path, exist_ok=True)

            with open(f'{metrics_path}/metrics_sl{self._his_window}_pl{self._n_step_ahead}.json', 'w') as f:
                json.dump(metrics, f, indent=4)

            with open(f'{metrics_path}/results_sl{self._his_window}_pl{self._n_step_ahead}.pkl', 'wb') as f:
                pickle.dump(results, f)

        return metrics