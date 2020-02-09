#!/Users/el/myenv/bin/python
"""
Test Neural Network with one hidden layer
"""

from typing import List

import numpy
import pandas
import structlog

ACTIVATION = "elu"
EPOCHS = 15000

activation = getattr(__import__("activation_functions", fromlist=[ACTIVATION]), ACTIVATION)
activation_prime = getattr(
    __import__("activation_functions", fromlist=[ACTIVATION + "_prime"]), ACTIVATION + "_prime"
)

numpy.random.seed(0)


class NeuralNetwork:
    """
    A neural network with an arbitrary number of hidden layers
    """

    def __init__(
        self,
        shape: List[int],
        learning_rate: float = 10e-6,
        loss_function=lambda x: x ** 2,
        name="learning-machine",
        has_bias=False,
    ):
        """
        Instantiate the model
        """
        self.weights = {
            idx: numpy.random.randn(shape[idx] + (has_bias if idx == 0 else 0), shape[idx + 1])
            for idx in range(len(shape) - 1)
        }

        self.deltas = {}
        self.has_bias = has_bias
        self.learning_rate = learning_rate
        self.logger = structlog.get_logger(name)
        self.loss = 0
        self.loss_function = loss_function
        self.n_layers = len(shape) - 1
        self.values = {level: {"input": None, "output": None} for level in range(len(shape) - 1)}

    def forwards(self, X):
        """
        Feed the data forward
        """
        if self.has_bias:
            X = numpy.append(X, 1)

        _in = X

        for layer_idx in range(self.n_layers):
            layer_input = numpy.dot(_in, self.weights[layer_idx])
            layer_output = activation(layer_input)

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
                error = numpy.dot(self.deltas[layer_idx + 1], self.weights[layer_idx + 1].T)

            delta = error * activation_prime(self.values[layer_idx]["input"])
            self.weights[layer_idx] += self.learning_rate * delta
            self.deltas[layer_idx] = delta

    def scalar(self, value):
        if isinstance(value, (int, float)):
            return value
        return self.scalar(value[0])

    def process(self, X, y):
        """
        Feed forward and backpropagate the data
        """
        self.forwards(X)
        self.backwards(X, y)

    def train(self, epochs):
        """
        Train the model
        """
        data = pandas.read_csv("../../../data/basic_data.csv", index_col="Unnamed: 0")
        for epoch in range(epochs):
            for _, row in data.iterrows():
                X = numpy.array([row[:-1]])
                y = numpy.array([row[-1]])
                self.process(X, y)
            self.logger.info(f"Mean Loss: {self.loss/data.shape[0]}")
            self.loss = 0


if __name__ == "__main__":

    nn = NeuralNetwork([3, 2, 1], has_bias=True, loss_function=abs, learning_rate=0.000005)
    nn.train(epochs=EPOCHS)
