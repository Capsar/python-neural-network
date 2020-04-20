# python-neural-network
A simple fully connected feed forward neural network written in python from scratch using numpy. It is possible to have multiple hidden layers, change the amount of neurons per layer &amp; have a different activation function per layer.

Written in python 3.7.7

If you have any tips on how to imporve performace, let me know!

```
import numpy as np
from numba.experimental import jitclass
from numba import types, typed
```

```
data_input = np.load("data/ci_inputs.npy")
data_output = np.load("data/ci_outputs.npy")

print("Begin compiling!")
begin_time = time.time_ns()
compile_nn = make_neural_network(layer_sizes=[data_input.shape[1], data_output.shape[1]], layer_activations=["sigmoid"])
compile_nn.train(data_input[:1], data_output[:1], data_input[1: 2], data_output[1: 2])
end_time = time.time_ns()
print("Compile time:", (end_time-begin_time) / 1e9)

for i in range(10):

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
    print("Seed:", random_seed, "Epochs:", epochs, "Time:", (end_time-begin_time)/1e9, "Accuracy:", accuracy_test, "Tr:", train_mse, "V:", current_mse, "T:", test_mse)
```
