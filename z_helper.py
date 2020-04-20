import numpy as np
from numba import njit


def import_from_csv(path, data_type):
    return np.genfromtxt(path, dtype=data_type, delimiter=',')


def class_to_array(maximum_class, x):
    data = np.zeros(maximum_class)
    data[x-1] = 1
    return data


def kfold(k, data, seed=99):
    np.random.seed(seed)
    data = np.random.permutation(data)
    fold_size = int(len(data) / k)
    return data[fold_size*2:], data[:fold_size], data[fold_size:fold_size*2]


@njit
def sigmoid(x, derivative):
    if derivative:
        return x * (1.0 - x)
    else:
        return 1.0 / (1.0 + np.exp(-x))


@njit
def relu(x, derivative):
    if derivative:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    else:
        return np.maximum(0, x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
