import numpy as np
import numpy_neural_network_2 as nn
import numpy_z_helper_2 as h
import time
np.set_printoptions(linewidth=200)

data_input = np.load("data/mnist_2_inputs.npy")
data_output = np.load("data/mnist_2_outputs.npy")

print(data_input.shape)
print(data_output.shape)

print("Begin compiling!")
np.random.seed(420)
begin_time = time.time_ns()
compiled_nn_values = nn.make_neural_network(layer_sizes=[data_input.shape[1], data_output.shape[1]], layer_activations=[h.sigmoid])
print("Created Neural network lists:", (time.time_ns()-begin_time) / 1e9)
nn.train_auto(data_input[:10], data_output[:10], data_input[10: 15], data_output[10: 15], 5, 500, compiled_nn_values)
print("Compile time:", (time.time_ns()-begin_time) / 1e9)

total_accuracy = 0.0
total_time = 0.0
n = 5
batch_size = 16
max_epochs = 200
for i in range(n):

    random_seed = np.random.randint(10, 1010)
    np.random.seed(random_seed)

    train_input, validate_input, test_input = h.kfold(7, data_input, random_seed)
    train_output, validate_output, test_output = h.kfold(7, data_output, random_seed)
    nn_values = nn.make_neural_network(layer_sizes=[train_input.shape[1], train_output.shape[1]], layer_activations=[h.softmax_2])

    begin_time = time.time_ns()
    epochs, current_mse = nn.train_auto(train_input, train_output, validate_input, validate_output, batch_size, max_epochs, nn_values)
    end_time = time.time_ns()

    train_mse = nn.calculate_MSE(train_input, train_output, nn_values)
    test_mse = nn.calculate_MSE(test_input, test_output, nn_values)
    accuracy_test = nn.evaluate(test_input, test_output, nn_values)

    total_accuracy += accuracy_test
    total_time += (end_time-begin_time)/1e9
    print("Seed:", random_seed, "Batch_size:", batch_size, "Epochs:", epochs, "Time:", (end_time-begin_time)/1e9, "Accuracy:", accuracy_test, "Tr:", train_mse, "V:", current_mse, "T:", test_mse)
print("Average Accuracy:", total_accuracy / n, "Average Time:", total_time / n)
