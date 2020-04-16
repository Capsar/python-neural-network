import numpy as np
import z_helper as h

class NeuralNetwork:

    def __init__(self, layer_sizes, layer_activations, learning_rate=0.1, low=-2, high=2):
        assert len(layer_sizes) >= 2
        assert len(layer_sizes)-1 == len(layer_activations)

        # Initialize weights between every neuron in all adjacent layers.
        self.weights = [h.random_np(low, high, (layer_sizes[i-1], layer_sizes[i])) for i in range(1, len(layer_sizes))]
        # Initialize biases for every neuron in all layers
        self.biases = np.array([h.random_np(low, high, layer_sizes[i]).reshape(-1, 1) for i in range(1, len(layer_sizes))])
        # Initialize empty list of output of every neuron in all layers.
        self.layer_outputs = [np.zeros(layer_sizes[i]).reshape(-1, 1) for i in range(len(layer_sizes))]

        self.layer_activations = layer_activations
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

    def calculate_output(self, input_data):
        input_data = np.array(input_data).reshape(-1, 1)
        assert len(input_data) == self.layer_sizes[0]
        num_calculations = len(self.weights)

        y = input_data
        self.layer_outputs[0] = y

        for i in range(num_calculations):
            y = h.activation(self.layer_activations[i])(np.dot(self.weights[i].T, y) + self.biases[i])
            self.layer_outputs[i+1] = y

        return y

    def train(self, input_data, desired_output_data):
        input_data = np.array(input_data).reshape(-1, 1)
        desired_output_data = np.array(desired_output_data).reshape(-1, 1)
        assert len(input_data) == self.layer_sizes[0]
        assert len(desired_output_data) == self.layer_sizes[-1]
        self.calculate_output(input_data)

        error = (desired_output_data - self.layer_outputs[-1]) * h.derivative(self.layer_activations[-1])(self.layer_outputs[-1])
        self.weights[-1] += (self.learning_rate * self.layer_outputs[-2] * error.T)
        self.biases[-1] += self.learning_rate * error

        for i in reversed(range(len(self.weights)-1)):
            error = np.dot(self.weights[i+1], error) * h.derivative(self.layer_activations[i])(self.layer_outputs[i+1])
            self.weights[i] += (self.learning_rate * self.layer_outputs[i] * error.T)
            self.biases[i] += self.learning_rate * error
    
    def calculate_SSE(self, input_data, desired_output_data):
        input_data = np.array(input_data).reshape(-1, 1)
        desired_output_data = np.array(desired_output_data).reshape(-1, 1)
        assert len(input_data) == self.layer_sizes[0]
        assert len(desired_output_data) == self.layer_sizes[-1]
        return np.sum(np.power(desired_output_data - self.calculate_output(input_data), 2))

    def print_weights_and_biases(self):
        print(self.weights)
        print(self.biases)


np.set_printoptions(linewidth=200)
for i in range(5):
    random_seed = np.random.randint(10, 1010)
    np.random.seed(random_seed)

    data_input = h.import_from_csv("data/features.txt", float)
    data_output = h.import_from_csv("data/targets.txt", int)
    data_output = np.array([h.class_to_array(np.amax(data_output), x) for x in data_output])

    train_input, validate_input, test_input = h.kfold(4, data_input, random_seed)
    train_output, validate_output, test_output = h.kfold(4, data_output, random_seed)

    nn = NeuralNetwork(layer_sizes=[10, 15, 7], layer_activations=["sigmoid", "sigmoid"])

    previous_mse = 1
    current_mse = 0
    epochs = 0
    while(current_mse < previous_mse):
        previous_mse = h.calculate_MSE(nn, validate_input, validate_output)
        for i in range(len(train_input)):
            nn.train(train_input[i], train_output[i])
        current_mse = h.calculate_MSE(nn, validate_input, validate_output)
        
        epochs += 1
        # if epochs % 10 == 0: print("Epoch: " + str(epochs) + " MSE: " + str(current_mse))


    train_mse = h.calculate_MSE(nn, train_input, train_output)
    test_mse = h.calculate_MSE(nn, test_input, test_output)
    print("Random_Seed: "  + str(random_seed) + " Epochs: " + str(epochs) + " Tr: " + str(train_mse) + " V: " + str(current_mse) + " T: " + str(test_mse))
