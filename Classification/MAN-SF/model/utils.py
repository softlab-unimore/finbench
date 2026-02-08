import numpy as np
import scipy
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data_1(graph):
    adj = scipy.sparse.coo_matrix(graph, dtype=np.float32)
    adj = normalize_adj(adj + scipy.sparse.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(row_sum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = scipy.sparse.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
