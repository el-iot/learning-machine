"""
Test Neural Network with one hidden layer
"""

from typing import List

import numpy
import pandas
import structlog

from . import activation_functions

numpy.random.seed(0)


class NeuralNetwork:
    """
    A neural network with an arbitrary number of hidden layers
    """

    def __init__(
        self,
        shape: List[int],
        activation="elu",
        has_bias=False,
        learning_rate: float = 10e-6,
        loss_function=lambda x: x ** 2,
        convergence_tolerance=0.995,
        name="learning-machine",
    ):
        """
        Instantiate the model
        """
        self.weights = {
            idx: numpy.random.randn(
                shape[idx] + (has_bias if idx == 0 else 0), shape[idx + 1]
            )
            for idx in range(len(shape) - 1)
        }

        self.deltas = {}
        self.has_bias = has_bias
        self.learning_rate = learning_rate
        self.logger = structlog.get_logger(name)
        self.loss = 0
        self.previous_loss = None
        self.loss_function = loss_function
        self.n_layers = len(shape) - 1
        self.values = {
            level: {"input": None, "output": None} for level in range(len(shape) - 1)
        }

        self.convergence_tolerance = convergence_tolerance
        self.activation = getattr(activation_functions, activation)
        self.activation_prime = getattr(activation_functions, activation + "_prime")

    def forwards(self, X):
        """
        Feed the data forward
        """
        if self.has_bias:
            X = numpy.append(X, 1)

        _in = X

        for layer_idx in range(self.n_layers):
            layer_input = numpy.dot(_in, self.weights[layer_idx])
            layer_output = self.activation(layer_input)

            self.values[layer_idx]["input"] = layer_input
            self.values[layer_idx]["output"] = _in = layer_output

    def backwards(self, X, y):
        """
        Backpropagate
        """
        _out = y
        for layer_idx in [*range(self.n_layers)][::-1]:

            if layer_idx == self.n_layers - 1:
                error = _out - self.values[layer_idx]["output"]
                self.loss += self.loss_function(self.scalar(error))
            else:
                error = numpy.dot(
                    self.deltas[layer_idx + 1], self.weights[layer_idx + 1].T
                )

            delta = error * self.activation_prime(self.values[layer_idx]["input"])
            self.weights[layer_idx] += self.learning_rate * delta
            self.deltas[layer_idx] = delta

    def scalar(self, value):
        """
        Convert to a scalar
        """
        if isinstance(value, (int, float)):
            return value
        return self.scalar(value[0])

    def process(self, X, y):
        """
        Feed forward and backpropagate the data
        """
        self.forwards(X)
        self.backwards(X, y)

    def train(self, data_path, converge=True, epochs=100):
        """
        Train the model
        """
        data = pandas.read_csv(data_path)
        epoch = 0
        trained = False

        while not trained:

            if not converge and epoch > epochs:
                self.logger.info(f"Finished {epochs} epochs")
                trained = True
                continue

            for _, row in data.iterrows():
                X = numpy.array([row[:-1]])
                y = numpy.array([row[-1]])
                self.process(X, y)

            mean_loss = self.loss / data.shape[0]
            self.logger.info(f"Mean Loss: {mean_loss}")

            if (
                False
                and self.previous_loss is not None
                and self.loss / self.previous_loss > self.convergence_tolerance
            ):
                self.logger.info("Converged")
                trained = True
                continue

            epoch += 1
            self.previous_loss, self.loss = mean_loss, 0
