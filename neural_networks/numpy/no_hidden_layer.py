#!/usr/local/bin/python3

"""
Neural Network with no hidden layers
"""

import numpy
import pandas

ACTIVATION = "leaky_relu"
DEBUG = True
EPOCHS = 15000
BIAS = True

activation = getattr(__import__("activation_functions", fromlist=[ACTIVATION]), ACTIVATION)
activation_prime = getattr(
    __import__("activation_functions", fromlist=[ACTIVATION + "_prime"]), ACTIVATION + "_prime"
)

numpy.random.seed(0)


class NeuralNetwork:
    """
    Neural Network class
    """

    def __init__(self):
        """
        Initialise model
        """
        self.w0 = numpy.random.randn(3 + (1 if BIAS else 0), 1)
        self.learning_rate = 0.0000005
        self.learning_rate_depreciation = 1
        self.loss = []
        self.verbose = DEBUG

    def forwards(self, X, y):
        """
        Feed-forward input data, X with target y
        """
        X = numpy.append(X, 1) if BIAS else X
        self.output_layer_input = numpy.dot(X, self.w0)
        return activation(self.output_layer_input)

    def backwards(self, yhat, y):
        """
        Back Propagate
        """
        output_error = y - yhat
        output_delta = output_error * activation_prime(self.output_layer_input)

        self.w0 += self.learning_rate * output_delta
        self.loss += [abs(y - yhat) ** 2]

    def process(self, X, y):
        """
        Perform one feed-forward and backpropagation cycle
        """
        yhat = self.forwards(X, y)
        self.backwards(yhat, y)

    def train(self, epochs):
        """
        Train the model on basic_data.csv
        """
        data = pandas.read_csv("../../data/basic_data.csv", index_col="Unnamed: 0")
        for epoch in range(epochs):
            for _, row in data.iterrows():
                X = numpy.array(row[:-1])
                y = numpy.array([row[-1]])
                self.process(X, y)
            self._print(f"Loss: {sum(self.loss)/data.shape[0]}")
            self.loss = []
            self.learning_rate *= self.learning_rate_depreciation

    def _print(self, s):
        if self.verbose:
            print(s)


if __name__ == "__main__":

    nn = NeuralNetwork()
    nn.train(epochs=EPOCHS)


# TODO:
# separate run-time config into either commandline args OR a yaml file
# add custom loss functions