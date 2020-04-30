import numpy as np
from numba import cuda
from numba import njit, types, typed


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

# @cuda.jit
# def multiply_stride(a, b, c): 
#   s1, s2 = cuda.grid(2)
#   d1, d2 = cuda.gridsize(2)
#   for i1 in range(s1, a.shape[0], d1): 
#     for i2 in range(s2, b.shape[1], d2): 
#       the_sum = 0
#       for k in range(b.shape[0]): # or a.shape[1] 
#         the_sum += a[i1][k]*b[k][i2]
#       c[i1, i2] = the_sum


# @njit
# def multiply(a, b):
#     d_a = cuda.to_device(a)
#     d_b = cuda.to_device(b)
#     c = np.zeros((a.shape[0], b.shape[1]))
#     d_c = cuda.to_device(c)
#     multiply_stride[(1,), (2,2)](d_a, d_b, d_c)
#     print(d_c.copy_to_host())


@njit
def activation(x, ftype, derivative):
    if derivative:
        return x * (1.0 - x)
    else:
        return 1.0 / (1.0 + np.exp(-x))


@njit
def relu(x, derivative):
    if derivative:
        return np.where(x <= 0, 0, 1)
    else:
        return np.maximum(0, x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# aa = np.arange(1000*1000).reshape(1000,1000)
# bb = np.arange(1000*1000).reshape(1000,1000)

# multiply(aa, bb)