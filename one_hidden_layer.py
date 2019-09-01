#!/Users/el/myenv/bin/python

"""
Test Neural Network with one hidden layer
"""
import numpy
import pandas
from time import sleep

ACTIVATION = "elu"
DEBUG = False
BIAS = False
EPOCHS = 15000

activation = getattr(
    __import__("activation_functions", fromlist=[ACTIVATION]), ACTIVATION
)
activation_prime = getattr(
    __import__("activation_functions", fromlist=[ACTIVATION + "_prime"]),
    ACTIVATION + "_prime",
)

numpy.random.seed(0)


class NeuralNetwork:
    """
    A neural network with one hidden layer
    """

    def __init__(self):
        """
        Instantiate the model
        """
        self.w0 = numpy.random.randn(1 + (1 if BIAS else 0), 2)
        self.w1 = numpy.random.randn(2, 1)
        self.learning_rate = 0.0000005

        self.loss = []
        self.verbose = DEBUG

    def feedforward(self, X):
        """
        Feed the data forward
        """
        if BIAS:
            X = numpy.append(X, 1)

        self._print(f"X: {X}")
        self.h0_input = numpy.dot(X, self.w0)
        self._print(f"h0_input: {self.h0_input}\nw0: {self.w0}")
        self.h0_output = activation(self.h0_input)
        self._print(f"h0_output: {self.h0_output}")

        self.output_layer_input = numpy.dot(self.h0_output, self.w1)
        self._print(f"output_layer_output: {self.output_layer_input}")
        self.output = activation(self.output_layer_input)
        self._print(f"output: {self.output}")

    def backprop(self, X, y):
        """
        Backpropagate
        """
        output_error = y - self.output
        self._print(f"output_error: {output_error}")
        output_delta = output_error * activation_prime(self.output_layer_input)
        self._print(f"output_delta: {output_delta}")

        h0_error = numpy.dot(output_delta, self.w1.T)
        self._print(f"h0_error: {h0_error}")
        h0_delta = h0_error * activation_prime(self.h0_input)
        self._print(f"h0_delta: {h0_delta}")
        self.w0 += self.learning_rate * h0_delta
        self.w1 += self.learning_rate * output_delta
        self.loss += [abs(y - self.output) ** 2]

    def process(self, X, y):
        """
        Feed forward and backpropagate the data
        """
        self.feedforward(X)
        self.backprop(X, y)

    def train(self, epochs):
        """
        Train the model
        """
        data = pandas.read_csv("basic_data.csv", index_col="Unnamed: 0").head(100)
        for epoch in range(epochs):
            for _, row in data.iterrows():
                X = numpy.array([[row[-1]]])
                y = numpy.array([row[-1] + 1])
                self.process(X, y)
            print(f"Total Loss: {sum(self.loss)}")
            self.loss = []

    def _print(self, s):
        """
        Print if verbose
        """
        if self.verbose:
            print(s)


if __name__ == "__main__":

    nn = NeuralNetwork()
    nn.train(epochs=EPOCHS)
