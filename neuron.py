from generator import create_data_class, plot_points

import numpy as np
from matplotlib import pyplot as plt


class ActivationFunctions:
    class Heaviside:
        @staticmethod
        def __activation(x: float) -> float:
            if x >= 0.:
                return 1.
            return 0.

        activation = np.vectorize(__activation)

        @staticmethod
        def derivative(s: np.ndarray) -> np.ndarray[float]:
            return np.ones_like(s)

    class Logistic:
        beta = -1.

        def activation(self, s: np.ndarray[float]) -> np.ndarray[float]:
            return 1. / (1. + np.exp(self.beta * s))

        def derivative(self, s: np.ndarray[float]) -> np.ndarray[float]:
            log_func = self.activation(s)
            return (log_func * (1 - log_func)) * -1 * self.beta

    class Sin:
        @staticmethod
        def activation(s: np.ndarray[float]) -> np.ndarray[float]:
            return np.sin(s)

        @staticmethod
        def derivative(s: np.ndarray[float]) -> np.ndarray[float]:
            return np.cos(s)

    class TanH:
        @staticmethod
        def activation(s: np.ndarray[float]) -> np.ndarray[float]:
            return np.tanh(s)

        @staticmethod
        def derivative(s: np.ndarray[float]) -> np.ndarray[float]:
            return 1. - np.tanh(s) ** 2

    class Sign:
        @staticmethod
        def __activation(x: float) -> float:
            if x > 0.:
                return 1.
            elif x < 0.:
                return -1.
            return 0.

        activation = np.vectorize(__activation)

        @staticmethod
        def derivative(s: np.ndarray[float]) -> np.ndarray[float]:
            return np.ones_like(s)

    class ReLu:
        @staticmethod
        def __activation(x: float) -> float:
            if x > 0.:
                return x
            return 0.

        @staticmethod
        def __derivative(x: float) -> float:
            if x > 0.:
                return 1.
            return 0.

        activation = np.vectorize(__activation)
        derivative = np.vectorize(__derivative)

    class LeakyReLu:
        @staticmethod
        def __activation(x: float) -> float:
            if x >= 0.:
                return x
            return 0.01 * x

        @staticmethod
        def __derivative(x: float) -> float:
            if x >= 0.:
                return 1.
            return 0.01

        activation = np.vectorize(__activation)
        derivative = np.vectorize(__derivative)


class Neuron:
    """
    Class implementing an artificial neuron.
    """

    def __init__(self, dimensions: int, activation_class, bias: float):
        self.dimensions = dimensions
        self.bias = bias
        self.weights = np.ones((1, dimensions), dtype=float) * 0.5
        self.ActivationClass = activation_class()

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

        points = np.append(samples, labels, axis=1)
        np.random.default_rng().shuffle(points)

        sample_batches = points[:, :self.dimensions].reshape((-1, batch_size, self.dimensions))
        label_batches = points[:, -1].reshape((-1, batch_size, 1))

        return label_batches, sample_batches

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
        print(samples.shape)
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
                state = sample @ self.weights.T - self.bias

                result = self.ActivationClass.activation(state)

                error = label - result

                weights_change = np.sum(learning_rate
                                        * error
                                        * self.ActivationClass.derivative(state)
                                        * sample, axis=0)
                self.weights += weights_change


if __name__ == '__main__':
    interval_modes_0 = (
        (0., 5.),
        (12., 12.)
    )

    interval_st_devs_0 = (
        (0.5, 0.5),
        (1., 1.)
    )

    interval_modes_1 = (
        (15., 15.),
        (20., 20.)
    )

    interval_st_devs_1 = (
        (0.5, 0.5),
        (3., 4.)
    )

    data_points_0_ = create_data_class(
        modes_size=6,
        sample_size=500,
        mode_intervals=interval_modes_0,
        std_dev_intervals=interval_st_devs_0,
        seed=10
    )
    data_points_1_ = create_data_class(
        modes_size=6,
        sample_size=500,
        mode_intervals=interval_modes_1,
        std_dev_intervals=interval_st_devs_1,
        seed=15
    )

    samples_ = np.append(data_points_0_, data_points_1_, axis=0)
    labels_ = np.append(np.zeros((6 * 500, 1)), np.ones((6 * 500, 1)), axis=0)

    samples_norm = (samples_ - np.min(samples_)) / (np.max(samples_) - np.min(samples_))

    neuron = Neuron(2, ActivationFunctions.ReLu, 0.6)
    neuron.train(samples_, labels_, 100, learning_rate=0.0005, batch_size=1)
    print(neuron.weights)
    fig, ax = plot_points(data_points_0_, data_points_1_)

    max_x, min_x = np.max(samples_[:, 0]), np.min(samples_[:, 0])
    max_y, min_y = np.max(samples_[:, 1]), np.min(samples_[:, 1])
    y1, y2 = neuron.weights[0, 0] * min_x + neuron.weights[0, 1], neuron.weights[0, 0] * max_x + neuron.weights[0, 1]

    ax.fill_between((min_x, max_x, min_x), (y2, y1, y1), facecolor='yellow', alpha=0.3)
    ax.fill_between((max_x, min_x, max_x), (y1, y2, y2), facecolor='green', alpha=0.3)
    ax.plot([max_x, min_x], [y2, y1])
    #ax.set_xlim([min_x, max_x])
    #ax.set_ylim([min_y, max_y])

    #ax.axline((max_x, y2), (min_x, y1))
    plt.show()
