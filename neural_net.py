from generator import create_data_class, plot_points
from neuron import ActivationFunctions

import numpy as np
from matplotlib import pyplot as plt


class Normalizer:
    def __init__(self, base):
        self.base_values = {"max": np.max(base, axis=0), "min": np.min(base, axis=0)}

    def minmax(self, values):
        return (values - self.base_values["min"]) / (self.base_values["max"] - self.base_values["min"])

    def minmax_dim(self, values, dim):
        return (values - self.base_values["min"][dim]) / (self.base_values["max"][dim] - self.base_values["min"][dim])


class NeuralNet:
    """
    Class implementing an artificial neuron.

    weigths -
    [
        [2 * [0.5 for _ in range(neurons_0)]],
        [neurons_0 * [0.5 for _ in range(neurons_1)]],
        [neurons_1 * [0.5 for _ in range(neurons_2)]],
        ...
    """

    def __init__(self, layers: tuple[int, ...], dimensions: int, activation_classes: list[ActivationFunctions]):
        self.dimensions = dimensions
        self.bias = []
        self.weights = []
        dd = np.random.randint(0, 1000000000)
        print(dd)
        for neurons_num, neurons_next in zip((dimensions, *layers[:-1]), layers):
            np.random.seed(seed=620425)
            self.bias += [np.random.randn(1, neurons_next)]
            self.weights += [np.random.randn(neurons_num, neurons_next)]

        self.activation_classes = activation_classes

    def create_batches(self,
                       samples: np.ndarray,
                       labels: np.ndarray,
                       batch_size: int) -> tuple[np.ndarray, np.ndarray[np.ndarray[float, float]]]:
        """
        Divides input samples and labels into batches.

        :param samples: An array of samples.
        :param labels: An array of labels.
        :param batch_size: Defines the size of batches.
        :return: An tuple consisting of batched arrays of labels and samples.
        """
        randomized_indexes = np.arange(len(samples))
        np.random.default_rng().shuffle(randomized_indexes)
        samples = samples[randomized_indexes]
        labels = labels[randomized_indexes]

        sample_batches = samples.reshape((-1, batch_size, self.dimensions))
        label_batches = labels.reshape((-1, batch_size, 2))

        return label_batches, sample_batches

    def predict(self, points):
        state = points
        for weight_layer, bias, activation_class in zip(self.weights, self.bias, self.activation_classes):
            state = activation_class.activation(state @ weight_layer - bias)
        return np.argmax(state, axis=1)

    def train(self,
              samples: np.ndarray,
              labels: np.ndarray,
              epochs: int,
              learning_rate: float | tuple[float, float],
              batch_size: int = 50):
        """
        A method used for training the neuron.

        :param samples:
        :param labels:
        :param epochs:
        :param learning_rate:
        :param batch_size:
        :return:
        """
        if isinstance(learning_rate, (list, tuple)):
            is_learning_rate_variable = True
            rate_min = learning_rate[0]
            rate_max = learning_rate[1]
            i_max = samples.shape[0] // batch_size
        else:
            is_learning_rate_variable = False
            rate_min = None
            rate_max = None
            i_max = None

        for epoch in range(epochs):
            label_batches, sample_batches = self.create_batches(samples, labels, batch_size)
            for i, (sample, label) in enumerate(zip(sample_batches, label_batches)):

                if is_learning_rate_variable:
                    learning_rate = (
                            rate_min + (rate_max - rate_min) * (1 + np.cos(np.pi * i / i_max))
                    )

                #  forward
                neuron_states = []
                previous_state = sample
                for weight_layer, bias, activation_class in zip(self.weights, self.bias, self.activation_classes):
                    state = previous_state @ weight_layer - bias
                    previous_state = activation_class.activation(state)
                    neuron_states.append(previous_state)

                #  backwards
                error = (label - neuron_states[-1])
                inputs_array = [sample, *neuron_states[:-1]]

                weights = self.weights[1:]
                weights.append(np.diagflat([1 for _ in range(error.shape[1])]))

                for num, (neuron_state, inputs, weight, activation_class) in enumerate(
                        zip(neuron_states[::-1], inputs_array[::-1], weights[::-1], self.activation_classes),
                        start=1
                ):
                    error = error @ weight.T * activation_class.derivative(neuron_state)

                    weights_change = (learning_rate * error.T @ inputs) / batch_size
                    bias_change = (learning_rate * np.sum(error, axis=0, keepdims=True)) / batch_size

                    self.weights[-num] += weights_change.T
                    self.bias[-num] += bias_change
        return 1


if __name__ == '__main__':
    interval_modes_0 = (
        (0., 10.),
        (10., 20.)
    )

    interval_st_devs_0 = (
        (0.5, 0.5),
        (1., 1.)
    )

    interval_modes_1 = (
        (20., 25.),
        (30., 50.)
    )

    interval_st_devs_1 = (
        (0.5, 0.5),
        (1., 1.)
    )

    data_points_0_ = create_data_class(
        modes_size=6,
        sample_size=200,
        mode_intervals=interval_modes_0,
        std_dev_intervals=interval_st_devs_0,
        seed=6
    )
    data_points_1_ = create_data_class(
        modes_size=6,
        sample_size=200,
        mode_intervals=interval_modes_1,
        std_dev_intervals=interval_st_devs_1,
        seed=56
    )

    samples_ = np.append(data_points_0_, data_points_1_, axis=0)
    labels_ = np.append([1, 0] * np.ones((6 * 200, 1)), [0, 1] * np.ones((6 * 200, 1)), axis=0)

    normalizer = Normalizer(samples_)
    samples_norm = normalizer.minmax(samples_)

    neural_net = NeuralNet((12, 10, 2), 2, [ActivationFunctions.TanH for _ in range(3)])
    neural_net.train(samples_norm, labels_, 200, 0.0002, 10)

    fig, ax = plot_points(data_points_0_, data_points_1_)

    X, Y = np.meshgrid(np.linspace(normalizer.base_values["min"][0] - 1, normalizer.base_values["max"][0] + 1, 200),
                       np.linspace(normalizer.base_values["min"][1] - 1, normalizer.base_values["max"][1] + 1, 200))

    X_norm = normalizer.minmax_dim(X, 0)
    Y_norm = normalizer.minmax_dim(Y, 1)

    points_graph = np.append(X_norm.reshape((-1, 1)), Y_norm.reshape((-1, 1)), axis=1)
    prediction = neural_net.predict(points_graph)
    ax.contourf(X, Y, prediction.reshape(X.shape), alpha=0.25, cmap=plt.cm.RdYlBu)

    plt.show()

