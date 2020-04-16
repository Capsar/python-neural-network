import numpy as np
import csv
import random

def import_from_csv(path, data_type):
    return  np.genfromtxt(path, dtype=data_type, delimiter=',') 

def class_to_array(maximum_class, x):
    data = np.zeros(maximum_class)
    data[x-1] = 1
    return data 

def kfold(k, data, seed=99):
    np.random.seed(seed)
    np.random.shuffle(data)
    fold_size = int(len(data) / k)
    return data[:fold_size], data[fold_size:fold_size*2], data[fold_size*2:]

def calculate_MSE(nn, input_data, output_data):
    size = len(input_data)
    sum_error = 0
    for i in range(size):
        sum_error += nn.calculate_SSE(input_data[i], output_data[i])
    return sum_error / size

def random_np(low, high, size):
    assert low <= high
    return np.random.random(size)*(high-low) + low

def activation(s):
    if s == "relu":
        return relu
    if s == "sigmoid":
        return sigmoid
    else:
        return "Error"

def derivative(s):
    if s == "relu":
        return relu_derivative
    if s == "sigmoid":
        return sigmoid_derivative
    else:
        return "Error"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
