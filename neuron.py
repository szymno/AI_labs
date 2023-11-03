import numpy as np


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
        beta = 0.1

        def activation(self, s: np.ndarray[float]) -> np.ndarray[float]:
            return 1. / (1. + np.exp(self.beta * s))

        def derivative(self, s: np.ndarray[float]) -> np.ndarray[float]:
            log_func = self.activation(s)
            return log_func * (1 - log_func)

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
