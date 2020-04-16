# python-neural-network
A simple fully connected feed forward neural network written in python from scratch using numpy. It is possible to have multiple hidden layers, change the amount of neurons per layer &amp; have a different activation function per layer.

Written in python 3.7.7

If you have any tips on how to imporve performace, let me know!

```
import numpy as np
import z_helper as h
```

```
    random_seed = random.randint(10, 1010)
    np.random.seed(random_seed)

    data_input = h.import_from_csv("data/features.txt", float)
    data_output = h.import_from_csv("data/targets.txt", int)
    data_output = np.array([h.class_to_array(np.amax(data_output), x) for x in data_output])

    train_input, validate_input, test_input = h.kfold(4, data_input, random_seed)
    train_output, validate_output, test_output = h.kfold(4, data_output, random_seed)

    nn = NeuralNetwork(layer_sizes=[10, 15, 7], layer_activations=["sigmoid", "sigmoid"])

    # print("Beginning training")
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
```
