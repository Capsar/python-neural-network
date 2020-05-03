import numpy as np
from numba.experimental import jitclass
from numba import njit, types, typed, prange, typeof
import numpy_z_helper_2 as h
import time

from numba.core.errors import NumbaTypeSafetyWarning
import warnings

warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

spec = [
    ("layer_sizes", types.ListType(types.int64)),
    ("layer_activations", types.ListType(types.FunctionType(types.float64[:, ::1](types.float64[:, ::1], types.boolean)))),
    ("weights", types.ListType(types.float64[:, ::1])),
    ("biases", types.ListType(types.float64[::1])),
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
    # print(typeof(typed_layer_activations))

    # Initialize weights between every neuron in all adjacent layers.
    typed_weights = typed.List()
    for i in range(1, len(layer_sizes)):
        typed_weights.append(np.random.uniform(low, high, (layer_sizes[i-1], layer_sizes[i])))
    # print(typeof(typed_weights))

    # Initialize biases for every neuron in all layers
    typed_biases = typed.List()
    for i in range(1, len(layer_sizes)):
        typed_biases.append(np.random.uniform(low, high, (layer_sizes[i],)))
    # print(typeof(typed_biases))

    # Initialize empty list of output of every neuron in all layers.
    typed_layer_outputs = typed.List()
    for i in range(len(layer_sizes)):
        typed_layer_outputs.append(np.zeros((layer_sizes[i], 1)))
    # print(typeof(typed_layer_outputs))

    # typed_layer_sizes = layer_sizes
    # typed_layer_activations = layer_activations
    # typed_weights = [np.random.uniform(low, high, (layer_sizes[i-1], layer_sizes[i])) for i in range(1, len(layer_sizes))]
    # typed_biases = [np.random.uniform(low, high, (layer_sizes[i],)) for i in range(1, len(layer_sizes))]
    # typed_layer_outputs = [np.zeros((layer_sizes[i],1)) for i in range(len(layer_sizes))]

    typed_learning_rate = learning_rate
    return NeuralNetwork(typed_layer_sizes, typed_layer_activations, typed_weights, typed_biases, typed_layer_outputs, typed_learning_rate)


@njit
def calculate_output(input_data, nn):
    assert input_data.shape[1] == nn.layer_sizes[0]
    y = input_data
    for i in prange(len(nn.weights)):
        y = nn.layer_activations[i](np.dot(y, nn.weights[i]) + nn.biases[i], False)
    return y


@njit
def feed_forward_layers(input_data, nn):
    assert input_data.shape[1] == nn.layer_sizes[0]
    nn.layer_outputs[0] = input_data
    for i in prange(len(nn.weights)):
        nn.layer_outputs[i+1] = nn.layer_activations[i](np.dot(nn.layer_outputs[i], nn.weights[i]) + nn.biases[i], False)


@njit
def train_batch(input_data, desired_output_data, nn):
    feed_forward_layers(input_data, nn)
    error = (desired_output_data - nn.layer_outputs[-1]) * nn.layer_activations[-1](nn.layer_outputs[-1], True)

    temp_weights = typed.List()
    temp_biases = typed.List()
    # temp_weights = []
    # temp_biases = []

    # nn.weights[-1] += nn.learning_rate * np.dot(nn.layer_outputs[-2].T, error) / input_data.shape[0]
    # nn.biases[-1] += nn.learning_rate * h.np_mean(0, error)
    temp_weights.insert(0, nn.weights[-1] + nn.learning_rate * np.dot(nn.layer_outputs[-2].T, error) / input_data.shape[0])
    temp_biases.insert(0, nn.biases[-1] + nn.learning_rate * h.np_mean(0, error))

    length_weights = len(nn.weights)
    for i in range(1, length_weights):
        i = length_weights - i - 1
        error = np.dot(error, nn.weights[i+1].T) * nn.layer_activations[i](nn.layer_outputs[i+1], True)
        # nn.weights[i] += nn.learning_rate * np.dot(nn.layer_outputs[i].T, error) / input_data.shape[0]
        # nn.biases[i] += nn.learning_rate * h.np_mean(0, error)
        temp_weights.insert(0, nn.weights[i] + nn.learning_rate * np.dot(nn.layer_outputs[i].T, error) / input_data.shape[0])
        temp_biases.insert(0, nn.biases[i] + nn.learning_rate * h.np_mean(0, error))

    nn.weights = temp_weights
    nn.biases = temp_biases


@njit(parallel=True)
def calculate_MSE(input_data, desired_output_data, nn):
    assert input_data.shape[0] == desired_output_data.shape[0]
    sum_error = np.sum(np.power(desired_output_data - calculate_output(input_data, nn), 2))
    return sum_error / len(input_data)


@njit
def train_auto(train_input_data, train_desired_output_data, validate_input_data, validate_output_data, nn):
    previous_mse = 1.0
    current_mse = 0.0
    epochs = 0
    batch_size = 8
    while(current_mse < previous_mse):
    # while(epochs < 40):
        epochs += 1
        previous_mse = calculate_MSE(validate_input_data, validate_output_data, nn)
        b, e = 0, batch_size
        while(e + batch_size <= len(train_input_data)):
            train_batch(train_input_data[b:e], train_desired_output_data[b:e], nn)
            b += batch_size
            e += batch_size
        current_mse = calculate_MSE(validate_input_data, validate_output_data, nn)
    return epochs, current_mse


@njit(parallel=True)
def evaluate(input_data, desired_output_data, nn):
    output_max = h.np_argmax(1, calculate_output(input_data, nn))
    desired_output_max = h.np_argmax(1, desired_output_data)
    difference_output_max = output_max - desired_output_max
    correct = np.count_nonzero(difference_output_max == 0)
    return correct / input_data.shape[0]


@njit
def print_weights_and_biases(nn):
    weights = np.clip(nn.weights[0], 0.001, 0.999)
    biases = np.clip(nn.biases[0], 0.001, 0.999)
    print(weights)
    print(biases)
