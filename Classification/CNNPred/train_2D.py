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

from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

from load_data import construct_data_warehouse_2d, split_train_val_test, extract_sequences_2d
from tensorflow.keras import backend as K, callbacks


def get_metrics(preds, labels):

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
    precision_neg = precision((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    recall_neg = recall((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    f_posit = 2 * ((precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon()))
    f_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))

    return (f_posit + f_neg) / 2


def train(data_warehouse, checkpoint_dir, n_features, args):

    print('sequencing ...')
    cnn_train_data, cnn_train_target, _, _, _ = extract_sequences_2d(data_warehouse, args.seq_len)
    cnn_valid_data, cnn_valid_target, _, _, _ = extract_sequences_2d(data_warehouse, args.seq_len, idx=(2,3))

    filepath = join(checkpoint_dir, '2D-models/best-{}-{}-{}-{}-{}-{}.weights.h5'.format(
        args.epochs, args.seq_len, args.pred_len, args.number_filter, args.dropout, args.seed))

    print(' fitting model to target')
    model = Sequential()

    # layer 1
    model.add(Conv2D(args.number_filter[0], (1, n_features),
               activation='relu', input_shape=(args.seq_len, n_features, 1)))

    # layer 2
    model.add(Conv2D(args.number_filter[1], (3, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 1)))

    # layer 3
    model.add(Conv2D(args.number_filter[2], (3, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 1)))

    model.add(Flatten())
    model.add(Dropout(args.dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam', loss='mae', metrics=['acc', f1]) # run_eagerly=True

    best_model = callbacks.ModelCheckpoint(filepath, monitor='val_f1', verbose=1, save_best_only=True,
                                           save_weights_only=True, mode='max')

    model.fit(cnn_train_data, cnn_train_target, epochs=args.epochs, batch_size=128, verbose=1,
                    validation_data=(cnn_valid_data, cnn_valid_target), callbacks=[best_model])

    model.load_weights(filepath)

    return model


def prediction(data_warehouse, model, dates, args):
    cnn_test_data, cnn_test_target, tickers, last_dates, pred_dates = extract_sequences_2d(data_warehouse, args.seq_len, dates, idx=(4,5))
    overall_results = model.predict(cnn_test_data)
    test_pred = (overall_results > 0.5).astype(int)
    test_pred = np.concatenate(test_pred, axis=0).reshape(-1)
    return test_pred, cnn_test_target, tickers, last_dates, pred_dates, overall_results


def run_cnn_ann(data_warehouse, checkpoint_dir, n_features, dates, args):

    K.clear_session()
    model = train(data_warehouse, checkpoint_dir, n_features, args)
    preds, labels, tickers, last_date, pred_date, preds_prob = prediction(data_warehouse, model, dates, args)

    metrics = get_metrics(preds, labels)
    metrics = {k: float(v) for k, v in metrics.items()}

    results = {
        'metrics': metrics,
        'preds': preds_prob,
        'labels': np.expand_dims(labels, axis=1),
        'pred_date': pred_date,
        'last_date': last_date,
        'tickers': tickers
    }

    with open(f'{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(f'{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl', 'wb') as f:
        pickle.dump(results, f)

    return metrics


if __name__=='__main__':

    args = ArgumentParser()

    args.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    args.add_argument('--universe', type=str, default='sp500', help='Universe of stocks to use')
    args.add_argument('--model_name', type=str, default='CNNPRed', help='Name of the model to use')

    args.add_argument('--pred_len', type=int, default=5, help='Steps for future prediction')
    args.add_argument('--seq_len', type=int, default=60, help='Lookback length for the model')
    args.add_argument('--start_date', type=str, default='2021-06-01', help='Start date for the dataset')
    args.add_argument('--end_train_date', type=str, default='2021-12-31', help='End date for training set')
    args.add_argument('--start_valid_date', type=str, default='2022-06-01', help='Start date for validation set')
    args.add_argument('--end_valid_date', type=str, default='2022-12-31', help='End date for validation set')
    args.add_argument('--start_test_date', type=str, default='2023-06-01', help='Start date for the test set')
    args.add_argument('--end_date', type=str, default='2023-12-31', help='End date for the dataset')

    args.add_argument('--epochs', type=int, default=200, help='Number of epochs')
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

    metrics_path = f'./results2D/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split("-")[0]}'
    os.makedirs(metrics_path, exist_ok=True)

    checkpoint_dir = f'./checkpoints2D/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split("-")[0]}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    data_warehouse, n_stocks, n_features = construct_data_warehouse_2d(args)
    data_warehouse, dates = split_train_val_test(data_warehouse, args)

    test_metrics = run_cnn_ann(data_warehouse, checkpoint_dir, n_features, dates, args)

    shutil.rmtree(tmpdir)
    print('Test Metrics:\n', test_metrics)