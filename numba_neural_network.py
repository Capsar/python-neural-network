import numpy as np
import cupy as cp
from numba.experimental import jitclass
from numba import njit, types, typed, prange
import z_helper as h
import time


spec = [
    ("weights", types.ListType(types.float64[:, ::1])),
    ("biases", types.ListType(types.float64[:, ::1])),
    ("layer_outputs", types.ListType(types.float64[:, ::1])),
]
@jitclass(spec)
class NeuralNetwork:
    def __init__(self, weights, biases, layer_outputs):
        self.weights = weights
        self.biases = biases
        self.layer_outputs = layer_outputs

    def copy(self):
        return NeuralNetwork(self.weights.copy(), self.biases.copy(), self.layer_outputs.copy())


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
    return (typed_layer_sizes, typed_layer_activations, typed_learning_rate, NeuralNetwork(typed_weights, typed_biases, typed_layer_outputs))

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
    for i in prange(len(nn[3].weights)):
        y = h.activation(np.dot(nn[3].weights[i].T, y) + nn[3].biases[i], nn[1][i], False)
    return y


@njit
def feed_forward_layers(input_data, nn):
    assert len(input_data) == nn[0][0]
    nn[3].layer_outputs[0] = input_data
    for i in range(len(nn[3].weights)):
        nn[3].layer_outputs[i+1] = h.activation(np.dot(nn[3].weights[i].T, nn[3].layer_outputs[i]) + nn[3].biases[i], nn[1][i], False)


@njit
def train_single(input_data, desired_output_data, nn):
    assert len(input_data) == nn[0][0]
    assert len(desired_output_data) == nn[0][-1]
    feed_forward_layers(input_data, nn)

    error = (desired_output_data - nn[3].layer_outputs[-1]) * h.activation(nn[3].layer_outputs[-1], nn[1][-1], True)
    nn[3].weights[-1] += (nn[2] * nn[3].layer_outputs[-2] * error.T)
    nn[3].biases[-1] += nn[2] * error

    length_weights = len(nn[3].weights)
    for i in range(1, length_weights):
        i = length_weights - i - 1
        error = np.dot(nn[3].weights[i+1], error) * h.activation(nn[3].layer_outputs[i+1], nn[1][i], True)
        nn[3].weights[i] += (nn[2] * nn[3].layer_outputs[i] * error.T)
        nn[3].biases[i] += nn[2] * error
    return nn

@njit
def train_batch(input_data, desired_output_data, nn):
    # new_nns = typed.List()
    length_input_data = len(input_data)

    # print(1, nn[3].weights)

    for i in range(length_input_data):
        train_single(input_data[i], desired_output_data[i], (nn[0], nn[1], nn[2], nn[3].copy()) )

    # print(2, nn[3].weights)

    # length_updates = len(nn[3].weights)
    # for new_nn in new_nns:
    #     for i in range(length_updates):
    #         nn[3].weights[i] += new_nn[3].weights[i]
    #         nn[3].biases[i] += new_nn[3].biases[i]

    #     for i in range(length_updates+1):
    #         nn[3].layer_outputs[i] += new_nn[3].layer_outputs[i]

    # print(3, nn[3].weights)

    # for i in range(length_updates):
    #     nn[3].weights[i] /= length_input_data+1
    #     nn[3].biases[i] /= length_input_data+1
    # for i in range(length_updates+1):
    #     nn[3].layer_outputs[i] /= length_input_data+1

    # print(4, nn[3].weights)


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
def train_auto(batch_size, train_input_data, train_desired_output_data, validate_input_data, validate_output_data, nn):
    previous_mse = 1.0
    current_mse = 0.0
    epochs = 0
    length_train_data = len(train_input_data)
    n_iterations = int(length_train_data / batch_size)

    while(current_mse < previous_mse):
        epochs += 1
        previous_mse = calculate_MSE(validate_input_data, validate_output_data, nn)
        # train_batch(train_input_data, train_desired_output_data, nn)
        # for i in range(len(train_input_data)):
        #     train_single(train_input_data[i], train_desired_output_data[i], nn)
        for i in range(n_iterations):
            b, e = i*batch_size, i*batch_size+batch_size
            train_batch(train_input_data[b:e], train_desired_output_data[b:e], nn)
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
    print(nn[3].weights)
    print(nn[3].biases)
