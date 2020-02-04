#!/Users/el/myenv/bin/python
"""
Numpy Implementation of a neural network
"""

import json
from typing import List

import numpy


def convert(entry):
    """
    Convert an entry to a binary representation
    """
    return {"x": [1, 0], "o": [0, 1], "b": [0, 0], "win": 1, "loss": -1, "draw": 0}[
        entry.replace("\n", "")
    ]


def ternary_to_binary(ternary: List, length=67):
    """
    Converts a ternary list-representation of a number
    into the binary list-representation of that number
    """
    if isinstance(ternary, list):
        ternary = "".join([str(_) for _ in ternary])
    binary = bin(
        sum(
            [3 ** (len(ternary) - i - 1) * int(ternary[i]) for i in range(len(ternary))]
        )
    )[2:]
    return [int(x) for x in "0" * (length - len(binary)) + binary]


class ConnectAI:
    """
    Connect-Four AI
    Uses a sigmoid activation function
    """

    def __init__(self, name, epochs=20):
        """
        Initialise the model
        """

        self.name = name
        self.epochs = epochs
        self.input_size = 84 + 1  # bias neuron
        self.h0_size = 6
        self.h1_size = 3
        self.output_size = 1

        self.w1 = numpy.random.rand(self.input_size, self.h0_size)  # 84 x 4 tensor
        self.w2 = numpy.random.rand(self.h0_size, self.h1_size)  # 4 x 1 tensor
        self.w3 = numpy.random.rand(self.h1_size, self.output_size)  # 4 x 1 tensor
        self.loss = []

    def forward(self, X):
        """
        Forward Pass
        """
        self.h0_input = numpy.dot(X, self.w1)
        self.h0_output = self.sigmoid(self.h0_input)

        self.h1_input = numpy.dot(self.h0_output, self.w2)
        self.h1_output = self.sigmoid(self.h1_input)

        self.output_layer_input = numpy.dot(self.h1_output, self.w3)
        output = self.sigmoid(self.output_layer_input)  # aka output_layer_output
        return output

    def backward(self, X, y, output):
        """
        Back Propagate
        """
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoidPrime(output)

        self.h1_error = numpy.dot(self.output_delta, self.w3.T)
        self.h1_delta = self.h1_error * self.sigmoidPrime(self.h1_output)

        self.h0_error = numpy.dot(self.h1_delta, self.w2.T)
        self.h0_delta = self.h0_error * self.sigmoidPrime(self.h0_output)

        self.w1 += numpy.dot(X.T, self.h0_delta)
        self.w2 += numpy.dot(self.h0_output.T, self.h1_delta)
        self.w3 += numpy.dot(self.h1_output.T, self.output_delta)

        self.loss += [abs(self.output_error)]

    def serialize(self, instance, reverse=False):
        """
        Serialize a line into 0's and 1's
        """
        X = self.flatten(
            [convert(x) for x in instance.split(",")][:-1][:: (-1 if reverse else 1)]
        )
        y = convert(instance.split(",")[-1])
        return (numpy.array([X + [1]]), numpy.array([y]))  # bias neuron

    def sigmoid(self, s):
        """
        Sigmoid Activation Function
        """
        return 1 / (1 + numpy.exp(-s))

    def sigmoidPrime(self, s):
        """
        Derivative of the sigmoid activation function
        """
        return s * (1 - s)

    def flatten(self, x):
        """
        Flattens a list of nested lists
        """
        if not isinstance(x, list):
            return [x]
        return [w for v in x for w in self.flatten(v)]

    def train(self, path):
        """
        Train
        """
        reverse = False
        for epoch in range(self.epochs):
            with open(path, "r") as file:
                for line in file.readlines():
                    X, y = self.serialize(line, reverse=reverse)
                    o = self.forward(X)
                    self.backward(X, y, o)
            print(f"{epoch}: {numpy.array(self.loss).mean()}")
            self.loss = []  # reset the loss

    def save_weights(self):
        """
        Save model weights
        """
        with open(f"weights/{self.name}.json", "w") as file:
            json.dump({"w1": self.w1.tolist(), "w2": self.w2.tolist()}, file)

    def load_weights(self):
        """
        Load model weights
        """
        pass


if __name__ == "__main__":

    EPOCHS = 20
    connect = ConnectAI(f"epoch_{EPOCHS}", epochs=EPOCHS)
    connect.train("connect-4.data")
    connect.save_weights()
