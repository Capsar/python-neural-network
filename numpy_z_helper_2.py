import numpy as np
from numba import njit


def import_from_csv(path, data_type):
    return np.genfromtxt(path, dtype=data_type, delimiter=',')


def class_to_array(maximum_class, x):
    data = np.zeros(maximum_class) + 0.01
    data[x-1] = 0.99
    return data


def kfold(k, data, seed=99):
    np.random.seed(seed)
    data = np.random.permutation(data)
    fold_size = int(len(data) / k)
    return data[fold_size*2:], data[:fold_size], data[fold_size:fold_size*2]


@njit
def np_func(npfunc, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = npfunc(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = npfunc(arr[i, :])
    return result


@njit
def np_argmax(axis, arr):
    # return arr.argmax(axis=axis)
    return np_func(np.argmax, axis, arr)


@njit
def np_max(axis, arr):
    # return arr.max(axis=axis)
    return np_func(np.max, axis, arr)


@njit
def np_mean(axis, arr):
    # return arr.mean(axis=axis)
    return np_func(np.mean, axis, arr)

@njit('float64[:, ::1](float64[:, ::1], boolean)')
def sigmoid(x, derivative):
    if derivative:
        return x * (1.0 - x)
    else:
        return 1.0 / (1.0 + np.exp(-x))


@njit('float64[:, ::1](float64[:, ::1], boolean)')
def relu(x, derivative):
    if derivative:
        return np.where(x <= 0.0, 0.0, 1.0)
    else:
        return np.maximum(0.0, x)


@njit('float64[:, ::1](float64[:, ::1], boolean)')
def leaky_relu(x, derivative):
    if derivative:
        return np.where(x <= 0.0, -0.01*x, 1.0)
    else:
        return np.maximum(-0.01*x, x)


@njit('float64[:, ::1](float64[:, ::1], boolean)')
def softmax(x, derivative):
    e_x = np.exp(x - np.max(x))
    result = e_x / e_x.sum()
    return result


@njit('float64[:, ::1](float64[:, ::1], boolean)')
def softmax_2(x, derivative):
    tmp = x - np_max(1, x).reshape(-1, 1)
    exp_tmp = np.exp(tmp)
    return exp_tmp / exp_tmp.sum(axis=1).reshape(-1, 1)
