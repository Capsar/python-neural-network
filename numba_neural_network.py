import numpy as np
from numba.experimental import jitclass
from numba import njit, types, typed, prange
import z_helper as h
import time

from numba.core.errors import NumbaTypeSafetyWarning
import warnings

warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

spec = [
    ("layer_sizes", types.ListType(types.int64)),
    ("layer_activations", types.ListType(types.FunctionType(types.float64[:, ::1](types.float64[:, ::1], types.boolean)))),
    ("weights", types.ListType(types.float64[:, ::1])),
    ("biases", types.ListType(types.float64[:, ::1])),
    ("layer_outputs", types.ListType(types.float64[:, ::1])),
    ("learning_rate", types.float64),
]
@jitclass(spec)
class NeuralNetwork:
    def __init__(self, layer_sizes, layer_activations, weights, biases, layer_outputs, learning_rate):
        self.layer_sizes = layer_sizes
        self.layer_activations = layer_activations
        self.weights = weights
        self.biases = biases
        self.layer_outputs = layer_outputs
        self.learning_rate = learning_rate


def make_neural_network(layer_sizes, layer_activations, learning_rate=0.05, low=-2, high=2):
    for size in layer_sizes:
        assert size > 0

    # Initialize typed layer sizes list.
    typed_layer_sizes = typed.List()
    for size in layer_sizes:
        typed_layer_sizes.append(size)
    # print(typeof(typed_layer_sizes))

    # Initialie typed layer activation method strings list.
    prototype = types.FunctionType(types.float64[:, ::1](types.float64[:, ::1], types.boolean))
    typed_layer_activations = typed.List.empty_list(prototype)
    for activation in layer_activations:
        typed_layer_activations.append(activation)

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
    return NeuralNetwork(typed_layer_sizes, typed_layer_activations, typed_weights, typed_biases, typed_layer_outputs, typed_learning_rate)


@njit
def calculate_output(input_data, nn):
    assert len(input_data) == nn.layer_sizes[0]
    y = input_data
    for i in prange(len(nn.weights)):
        y = nn.layer_activations[i](np.dot(nn.weights[i].T, y) + nn.biases[i], False)
    return y


@njit
def feed_forward_layers(input_data, nn):
    assert len(input_data) == nn.layer_sizes[0]
    nn.layer_outputs[0] = input_data
    for i in prange(len(nn.weights)):
        nn.layer_outputs[i+1] = nn.layer_activations[i](np.dot(nn.weights[i].T, nn.layer_outputs[i]) + nn.biases[i], False)


@njit
def train_single(input_data, desired_output_data, nn):
    assert len(input_data) == nn.layer_sizes[0]
    assert len(desired_output_data) == nn.layer_sizes[-1]
    feed_forward_layers(input_data, nn)

    error = (desired_output_data - nn.layer_outputs[-1]) * nn.layer_activations[-1](nn.layer_outputs[-1], True)
    nn.weights[-1] += nn.learning_rate * nn.layer_outputs[-2] * error.T
    nn.biases[-1] += nn.learning_rate * error

    length_weights = len(nn.weights)
    for i in prange(1, length_weights):
        i = length_weights - i - 1
        error = np.dot(nn.weights[i+1], error) * nn.layer_activations[i](nn.layer_outputs[i+1], True)
        nn.weights[i] += nn.learning_rate * nn.layer_outputs[i] * error.T
        nn.biases[i] += nn.learning_rate * error
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
    print(nn.weights)
    print(nn.biases)
