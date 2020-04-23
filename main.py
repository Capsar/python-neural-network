import numpy as np
from numba.experimental import jitclass
from numba import types, typed
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

    return NeuralNetwork(typed_layer_sizes, typed_layer_activations, typed_weights, typed_biases, typed_layer_outputs, learning_rate, low, high)


spec = [
    ("layer_sizes", types.ListType(types.int64)),
    ("layer_activations", types.ListType(types.string)),
    ("weights", types.ListType(types.float64[:, ::1])),
    ("biases", types.ListType(types.float64[:, ::1])),
    ("layer_outputs", types.ListType(types.float64[:, ::1])),
    ("learning_rate", types.float64),
    ("low", types.int64),
    ("high", types.int64)
]
@jitclass(spec)
class NeuralNetwork:
    def __init__(self, layer_sizes, layer_activations, weights, biases, layer_outputs, learning_rate, low, high):
        assert len(layer_sizes) >= 2
        assert len(layer_sizes)-1 == len(layer_activations)

        self.layer_sizes = layer_sizes

        # Initialize list with activation functions per layer.
        self.layer_activations = layer_activations
        self.weights = weights
        self.biases = biases
        self.layer_outputs = layer_outputs

        self.learning_rate = learning_rate

    def calculate_output(self, input_data):
        assert len(input_data) == self.layer_sizes[0]

        y = input_data
        self.layer_outputs[0] = y

        for i in range(len(self.weights)):
            y = h.activation(np.dot(self.weights[i].T, y) + self.biases[i], self.layer_activations[i], False)
            self.layer_outputs[i+1] = y
        return y

    def train_single(self, input_data, desired_output_data):
        assert len(input_data) == self.layer_sizes[0]
        assert len(desired_output_data) == self.layer_sizes[-1]
        self.calculate_output(input_data)

        error = (desired_output_data - self.layer_outputs[-1]) * h.activation(self.layer_outputs[-1], self.layer_activations[-1], True)
        self.weights[-1] += (self.learning_rate * self.layer_outputs[-2] * error.T)
        self.biases[-1] += self.learning_rate * error

        length_weights = len(self.weights)
        for i in range(1, length_weights):
            i = length_weights - i - 1
            error = np.dot(self.weights[i+1], error) * h.activation(self.layer_outputs[i+1], self.layer_activations[i], True)
            self.weights[i] += (self.learning_rate * self.layer_outputs[i] * error.T)
            self.biases[i] += self.learning_rate * error

    def calculate_SSE(self, input_data, desired_output_data):
        assert input_data.shape[0] == self.layer_sizes[0]
        assert desired_output_data.shape[0] == self.layer_sizes[-1]
        return np.sum(np.power(desired_output_data - self.calculate_output(input_data), 2))

    def calculate_MSE(self, input_data, desired_output_data):
        assert input_data.shape[0] == desired_output_data.shape[0]
        size = input_data.shape[0]
        sum_error = 0
        for i in range(size):
            sum_error += self.calculate_SSE(input_data[i], desired_output_data[i])
        return sum_error / size

    def train(self, test_input_data, test_desired_output_data, validate_input_data, validate_output_data):
        previous_mse = 1.0
        current_mse = 0.0
        epochs = 0
        while(current_mse < previous_mse):
            epochs += 1
            previous_mse = self.calculate_MSE(validate_input_data, validate_output_data)
            for i in range(len(test_input_data)):
                self.train_single(test_input_data[i], test_desired_output_data[i])
            current_mse = self.calculate_MSE(validate_input_data, validate_output_data)
        return epochs, current_mse

    def evaluate(self, input_data, desired_output_data):
        corrects, wrongs = 0, 0
        for i in range(len(input_data)):
            output = self.calculate_output(input_data[i])
            output_max = output.argmax()
            desired_output_max = desired_output_data[i].argmax()
            if output_max == desired_output_max:
                corrects += 1
            else:
                wrongs += 1
        return corrects / (corrects + wrongs) 

    def print_weights_and_biases(self):
        print(self.weights)
        print(self.biases)


np.set_printoptions(linewidth=200)

data_input = np.load("data/ci_inputs.npy")
data_output = np.load("data/ci_outputs.npy")

print("Begin compiling!")
begin_time = time.time_ns()
compile_nn = make_neural_network(layer_sizes=[data_input.shape[1], data_output.shape[1]], layer_activations=["sigmoid"])
compile_nn.train(data_input[:1], data_output[:1], data_input[1: 2], data_output[1: 2])
end_time = time.time_ns()
print("Compile time:", (end_time-begin_time) / 1e9)
np.random.seed(420)

total_time = 0.0
n = 10
for i in range(n):

    random_seed = np.random.randint(10, 1010)
    np.random.seed(random_seed)

    train_input, validate_input, test_input = h.kfold(4, data_input, random_seed)
    train_output, validate_output, test_output = h.kfold(4, data_output, random_seed)

    nn = make_neural_network(layer_sizes=[train_input.shape[1], 20, train_output.shape[1]], layer_activations=["sigmoid", "sigmoid"])

    begin_time = time.time_ns()
    epochs, current_mse = nn.train(train_input, train_output, validate_input, validate_output)
    end_time = time.time_ns()

    train_mse = nn.calculate_MSE(train_input, train_output)
    test_mse = nn.calculate_MSE(test_input, test_output)

    accuracy_test = nn.evaluate(test_input, test_output)
    total_time += (end_time-begin_time)/1e9

    print("Seed:", random_seed, "Epochs:", epochs, "Time:", (end_time-begin_time)/1e9, "Accuracy:", accuracy_test, "Tr:", train_mse, "V:", current_mse, "T:", test_mse)
print("Average:", total_time / n)
