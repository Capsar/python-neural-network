import numpy as np
import z_helper as h
import time


class NeuralNetwork:
    def __init__(self, layer_sizes, layer_activations, learning_rate=0.1, low=-2, high=2):
        assert len(layer_sizes) >= 2
        assert len(layer_sizes)-1 == len(layer_activations)

        # Initialize weights between every neuron in all adjacent layers.
        self.weights = np.array([np.random.uniform(low, high, (layer_sizes[i-1], layer_sizes[i])) for i in range(1, len(layer_sizes))])
        # Initialize biases for every neuron in all layers

        self.biases = np.array([np.random.uniform(low, high, (layer_sizes[i], 1)) for i in range(1, len(layer_sizes))])
        # Initialize empty list of output of every neuron in all layers.
        self.layer_outputs = np.array([np.zeros((layer_sizes[i], 1)) for i in range(len(layer_sizes))])

        self.layer_activations = layer_activations
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

    def calculate_output(self, input_data):
        assert len(input_data) == self.layer_sizes[0]

        y = input_data
        self.layer_outputs[0] = y

        for i in range(self.weights.shape[0]):
            y = self.layer_activations[i](np.dot(self.weights[i].T, y) + self.biases[i], False)
            self.layer_outputs[i+1] = y
        return y

    def train(self, input_data, desired_output_data):
        assert input_data.shape[0] == self.layer_sizes[0]
        assert desired_output_data.shape[0] == self.layer_sizes[-1]
        self.calculate_output(input_data)

        error = (desired_output_data - self.layer_outputs[-1]) * self.layer_activations[-1](self.layer_outputs[-1], True)
        self.weights[-1] += (self.learning_rate * self.layer_outputs[-2] * error.T)
        self.biases[-1] += self.learning_rate * error

        for i in reversed(range(self.weights.shape[0]-1)):
            error = np.dot(self.weights[i+1], error) * self.layer_activations[i](self.layer_outputs[i+1], True)
            self.weights[i] += (self.learning_rate * self.layer_outputs[i] * error.T)
            self.biases[i] += self.learning_rate * error

    def calculate_SSE(self, input_data, desired_output_data):
        assert input_data.shape[0] == self.layer_sizes[0]
        assert desired_output_data.shape[0] == self.layer_sizes[-1]
        return np.sum(np.power(desired_output_data - self.calculate_output(input_data), 2))

    def calculate_MSE(self, input_data, output_data):
        assert input_data.shape[0] == output_data.shape[0]
        size = input_data.shape[0]
        sum_error = 0
        for i in range(size):
            sum_error += self.calculate_SSE(input_data[i], output_data[i])
        return sum_error / size

    def print_weights_and_biases(self):
        print(self.weights)
        print(self.biases)


np.set_printoptions(linewidth=200)

data_input = h.import_from_csv("data/features.txt", float)
data_output = h.import_from_csv("data/targets.txt", int)
data_output = np.array([h.class_to_array(np.amax(data_output), x) for x in data_output])

data_input = data_input.reshape((len(data_input), -1, 1))
data_output = data_output.reshape((len(data_input), -1, 1))

for i in range(4):
    random_seed = 10
    np.random.seed(random_seed)

    train_input, validate_input, test_input = h.kfold(4, data_input, random_seed)
    train_output, validate_output, test_output = h.kfold(4, data_output, random_seed)

    nn = NeuralNetwork(layer_sizes=[10, 15, 7], layer_activations=[h.sigmoid, h.sigmoid])
    # test_mse = nn.calculate_MSE(test_input, test_output)
    # print("TEST MSE:", test_mse)

    previous_mse = 1
    current_mse = 0
    epochs = 0
    begin_time = time.time_ns()
    while(current_mse < previous_mse):
        epochs += 1
        previous_mse = nn.calculate_MSE(validate_input, validate_output)
        for i in range(len(train_input)):
            nn.train(train_input[i], train_output[i])
        current_mse = nn.calculate_MSE(validate_input, validate_output)
    end_time = time.time_ns()

    train_mse = nn.calculate_MSE(train_input, train_output)
    test_mse = nn.calculate_MSE(test_input, test_output)
    print("Seed:", random_seed, "Epochs:", epochs, "Time:", (end_time-begin_time)/1e9, "Tr:", train_mse, "V:", current_mse, "T:", test_mse)
