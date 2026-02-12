import json
import os
import pickle
import shutil
import tempfile
from argparse import ArgumentParser
from os.path import join

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D

from load_data import construct_data_warehouse_3d
from tensorflow.keras import backend as K, callbacks


def get_metrics(preds, labels):
    preds = np.concatenate(preds, axis=0).reshape(-1)
    labels = np.concatenate(labels, axis=0).reshape(-1)

    metrics = {
        'F1_macro': f1_score(labels, preds, average='macro'),
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds, average='macro'),
        'Recall': recall_score(labels, preds, average='macro'),
        'MCC': matthews_corrcoef(labels, preds)
    }
    return metrics


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """

        y_true = K.cast(y_true, K.floatx())
        y_pred = K.squeeze(y_pred, axis=-1)

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.squeeze(y_pred, axis=-1)

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision_pos = precision(y_true, y_pred)
    recall_pos = recall(y_true, y_pred)
    precision_neg = precision((K.ones_like(y_true)-y_true), (K.ones_like(y_pred)-K.clip(y_pred, 0, 1)))
    recall_neg = recall((K.ones_like(y_true)-y_true), (K.ones_like(y_pred)-K.clip(y_pred, 0, 1)))
    f_posit = 2*((precision_pos*recall_pos)/(precision_pos+recall_pos+K.epsilon()))
    f_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))

    return (f_posit + f_neg) / 2


def prediction(test_dataset, model):
    preds_prob = model.predict(test_dataset)
    preds = (preds_prob > 0.5).astype(int)
    return preds, preds_prob


def run_cnn_ann_3d(dataset, n_stocks, n_features, checkpoint_dir, dates, tickers, metrics_path, args):

    preds = []
    preds_prob = []

    for i in range(n_stocks):
        train_labels = dataset[1][:, i]
        valid_labels = dataset[3][:, i]

        K.clear_session()
        filepath = join(checkpoint_dir, '3D-models/best-{}-{}-{}-{}-{}-{}.weights.h5'.format(
            args.epochs, args.seq_len, args.pred_len, args.number_filter, args.dropout, args.seed))

        # If the trained model doesn't exit, it is trained
        print('fitting model')
        model = Sequential()

        #layer 1
        model.add(Conv2D(args.number_filter[0], (1, 1), activation='relu', input_shape=(n_stocks, args.seq_len, n_features), data_format='channels_last'))
        #layer 2
        model.add(Conv2D(args.number_filter[1], (n_stocks, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(1, 2)))

        #layer 3
        model.add(Conv2D(args.number_filter[2], (1, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(1, 2)))

        model.add(Flatten())
        model.add(Dropout(args.dropout))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='Adam', loss='mae', metrics=['acc',f1])

        best_model = callbacks.ModelCheckpoint(filepath, monitor='val_f1', verbose=0, save_best_only=True,
                                               save_weights_only=True, mode='max')

        model.fit(dataset[0], train_labels, epochs=args.epochs, batch_size=128, verbose=1,
                callbacks=[best_model], validation_data=(dataset[2], valid_labels))

        model.load_weights(filepath)

        pred, pred_prob = prediction(dataset[4], model)
        preds.append(pred)
        preds_prob.append(pred_prob)

    preds = np.array(preds).squeeze(-1).swapaxes(0,1)
    preds_prob = np.array(preds_prob).swapaxes(0,1)

    metrics = get_metrics(preds, dataset[5])
    metrics = {k: float(v) for k, v in metrics.items()}

    labels = [np.expand_dims(dataset[5][i], axis=1) for i in range(dataset[5].shape[0])]
    preds_prob = [np.expand_dims(preds_prob[i], axis=1) for i in range(preds_prob.shape[0])]

    results = {
        'metrics': metrics,
        'preds': preds_prob,
        'labels': labels,
        'pred_date': dates['pred_date'],
        'last_date': dates['last_date'],
        'tickers': [tickers] * len(preds)
    }

    with open(f'{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(f'{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl', 'wb') as f:
        pickle.dump(results, f)

    return metrics


if __name__=='__main__':

    args = ArgumentParser()

    args.add_argument('--data_path', type=str, default='../../Evaluation/data', help='Path to the dataset')
    args.add_argument('--universe', type=str, default='sp500', help='Universe of stocks to use')
    args.add_argument('--model_name', type=str, default='CNNPRed', help='Name of the model to use')

    args.add_argument('--pred_len', type=int, default=5, help='Steps for future prediction')
    args.add_argument('--seq_len', type=int, default=60, help='Lookback length for the model')
    args.add_argument('--start_date', type=str, default='2021-06-01', help='Start date for the dataset')
    args.add_argument('--end_train_date', type=str, default='2021-12-31', help='End date for training set')
    args.add_argument('--start_valid_date', type=str, default='2022-06-01', help='Start date for validation set')
    args.add_argument('--end_valid_date', type=str, default='2022-12-31', help='End date for validation set')
    args.add_argument('--start_test_date', type=str, default='2023-01-01', help='Start date for the test set')
    args.add_argument('--end_date', type=str, default='2023-12-31', help='End date for the dataset')

    args.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    args.add_argument('--number_filter', type=int, nargs='+', default=[8, 8, 8])
    args.add_argument('--dropout', type=float, default=0.1)
    args.add_argument('--seed', type=int, default=42)

    args = args.parse_args()

    current_path = os.getcwd()
    base_tmp = os.path.join(current_path, "tmp")
    os.makedirs(base_tmp, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix="tf_run_", dir=base_tmp)
    os.environ["TMPDIR"] = tmpdir

    print("Temp directory:", tmpdir)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    tf.random.set_seed(args.seed)

    metrics_path = f'./results3D/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split("-")[0]}'
    os.makedirs(metrics_path, exist_ok=True)

    checkpoint_dir = f'./checkpoints3D/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split("-")[0]}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    splitted_dataset, tickers, dates = construct_data_warehouse_3d(args)
    n_features = splitted_dataset[0].shape[-1]
    n_stocks = splitted_dataset[0].shape[1]

    test_metrics = run_cnn_ann_3d(splitted_dataset, n_stocks, n_features, checkpoint_dir, dates, tickers, metrics_path, args)

    shutil.rmtree(tmpdir)
    print('Test Metrics:\n', test_metrics)