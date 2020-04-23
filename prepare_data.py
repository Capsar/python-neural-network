import numpy as np
import z_helper as h


def prepare_mnist_data():
    print(1)
    mnist_dataset = h.import_from_csv("data/mnist_train.csv", int)
    print(2)
    data_input = mnist_dataset[:, 1:].astype(float)
    print(3)
    data_input = data_input * (0.99 / 255.0) + 0.01
    print(4)
    data_output = mnist_dataset[:, :1].astype(int)
    print(5)

    data_output = np.array([h.class_to_array(np.amax(data_output), x) for x in data_output])
    print(6)
    data_input = data_input.reshape((len(data_input), -1, 1))
    print(7)
    data_output = data_output.reshape((len(data_output), -1, 1))
    print(8)

    np.save("data/mnist_inputs", data_input)
    np.save("data/mnist_outputs", data_output)


def prepare_ci_data():
    data_input = h.import_from_csv("data/ci_features.txt", float)
    data_output = h.import_from_csv("data/ci_targets.txt", int)

    data_output = np.array([h.class_to_array(np.amax(data_output), x) for x in data_output])
    data_input = data_input.reshape((len(data_input), -1, 1))
    data_output = data_output.reshape((len(data_output), -1, 1))
    np.save("data/ci_inputs", data_input)
    np.save("data/ci_outputs", data_output)

prepare_mnist_data()