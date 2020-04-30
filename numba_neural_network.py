import numpy as np
# import cupy as cp
from numba import njit, types, typed, prange
import z_helper as h
import time


def make_neural_network(layer_sizes, layer_activations, learning_rate=0.05, low=-2, high=2):

    # Initialize typed layer sizes list.
    typed_layer_sizes = typed.List()
    for size in layer_sizes:
        typed_layer_sizes.append(size)
    # print(typeof(typed_layer_sizes))

    # Initialie typed layer activation method strings list.
    typed_layer_activations = typed.List()
    for activation in layer_activations:
        typed_layer_activations.append(activation)
    # print(typeof(typed_layer_activations))

    # Initialize weights between every neuron in all adjacent layers.
    typed_weights = typed.List()
    for i in range(1, len(layer_sizes)):
        typed_weights.append(np.random.uniform(low, high, (layer_sizes[i-1], layer_sizes[i])))
    # print(typeof(typed_weights))

    # Initialize biases for every neuron in all layers
    typed_biases = typed.List()
    for i in range(1, len(layer_sizes)):
        typed_biases.append(np.random.uniform(low, high, (layer_sizes[i], 1)))
    # print(typeof(typed_biases))

    # Initialize empty list of output of every neuron in all layers.
    typed_layer_outputs = typed.List()
    for i in range(len(layer_sizes)):
        typed_layer_outputs.append(np.zeros((layer_sizes[i], 1)))
    # print(typeof(typed_layer_outputs))

    typed_learning_rate = learning_rate
    return (typed_layer_sizes, typed_layer_activations, typed_weights, typed_biases, typed_layer_outputs, typed_learning_rate)

# typed_layer_sizes = 0
# typed_layer_activations = 1
# typed_weights = 2
# typed_biases = 3
# typed_layer_outputs = 4
# typed_learning_rate = 5


@njit
def calculate_output(input_data, nn):
    assert len(input_data) == nn[0][0]
    y = input_data
    for i in prange(len(nn[2])):
        y = h.activation(np.dot(nn[2][i].T, y) + nn[3][i], nn[1][i], False)
    return y


@njit
def feed_forward_layers(input_data, nn):
    assert len(input_data) == nn[0][0]
    nn[4][0] = input_data
    for i in range(len(nn[2])):
        nn[4][i+1] = h.activation(np.dot(nn[2][i].T, nn[4][i]) + nn[3][i], nn[1][i], False)


@njit
def train_single(input_data, desired_output_data, nn):
    assert len(input_data) == nn[0][0]
    assert len(desired_output_data) == nn[0][-1]
    feed_forward_layers(input_data, nn)

    error = (desired_output_data - nn[4][-1]) * h.activation(nn[4][-1], nn[1][-1], True)
    nn[2][-1] += (nn[5] * nn[4][-2] * error.T)
    nn[3][-1] += nn[5] * error

    length_weights = len(nn[2])
    for i in range(1, length_weights):
        i = length_weights - i - 1
        error = np.dot(nn[2][i+1], error) * h.activation(nn[4][i+1], nn[1][i], True)
        nn[2][i] += (nn[5] * nn[4][i] * error.T)
        nn[3][i] += nn[5] * error
    return nn


@njit(parallel=True)
def calculate_MSE(input_data, desired_output_data, nn):
    assert input_data.shape[0] == desired_output_data.shape[0]
    size = input_data.shape[0]
    sum_error = 0
    for i in prange(size):
        sum_error += np.sum(np.power(desired_output_data[i] - calculate_output(input_data[i], nn), 2))
    return sum_error / size


@njit
def train_epoch(train_input_data, train_desired_output_data, validate_input_data, validate_output_data, n_epochs, nn):
    previous_mse = 1.0
    current_mse = 0.0
    for e in range(n_epochs):
        for i in range(len(train_input_data)):
            train_single(train_input_data[i], train_desired_output_data[i], nn)
    current_mse = calculate_MSE(validate_input_data, validate_output_data, nn)
    return current_mse


@njit
def train_auto(train_input_data, train_desired_output_data, validate_input_data, validate_output_data, nn):
    previous_mse = 1.0
    current_mse = 0.0
    epochs = 0
    while(current_mse < previous_mse):
        epochs += 1
        previous_mse = calculate_MSE(validate_input_data, validate_output_data, nn)
        for i in range(len(train_input_data)):
            train_single(train_input_data[i], train_desired_output_data[i], nn)
        current_mse = calculate_MSE(validate_input_data, validate_output_data, nn)
    return epochs, current_mse


@njit(parallel=True)
def evaluate(input_data, desired_output_data, nn):
    corrects, wrongs = 0, 0
    for i in prange(len(input_data)):
        output = calculate_output(input_data[i], nn)
        output_max = output.argmax()
        desired_output_max = desired_output_data[i].argmax()
        if output_max == desired_output_max:
            corrects += 1
        else:
            wrongs += 1
    return corrects / (corrects + wrongs)


@njit
def print_weights_and_biases(nn):
    print(nn[2])
    print(nn[3])
