#!/Users/el/myenv/bin/python
"""
Numpy Implementation of a neural network
"""

from typing import List

import json
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
    Converts a ternary list-representation of a number into the binary list-representation of that number
    """
    if isinstance(ternary, list):
        ternary = "".join([str(_) for _ in ternary])
    binary = bin(sum([3 ** (len(ternary) - i - 1) * int(ternary[i]) for i in range(len(ternary))]))[
        2:
    ]
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
        self.input_size = 84
        self.hidden_size = 10
        self.output_size = 1

        self.w1 = numpy.random.rand(self.input_size, self.hidden_size)  # 84 x 4 tensor
        self.w2 = numpy.random.rand(self.hidden_size, self.output_size)  # 4 x 1 tensor
        self.loss = []

    def forward(self, X):
        """
        Forward Pass
        """
        self.z = numpy.dot(X, self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = numpy.dot(self.z2, self.w2)
        output = self.sigmoid(self.z3)
        return output

    def backward(self, X, y, output):
        """
        Back Propagate
        """
        self.output_error = y - output  # error in output
        self.loss += [abs(self.output_error)]
        self.output_delta = self.output_error * self.sigmoidPrime(
            output
        )  # derivative of sig to error
        self.z2_error = numpy.dot(self.output_delta, self.w2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.w1 += numpy.dot(X.T, self.z2_delta)
        self.w2 += numpy.dot(self.z2.T, self.output_delta)

    def serialize(self, instance, reverse=False):
        """
        Serialize a line into 0's and 1's
        """
        serialized_instance = self.flatten(
            [convert(x) for x in instance.split(",")][::(-1 if reverse else 1)]
        )
        return (numpy.array([serialized_instance[:-1]]), numpy.array([serialized_instance[-1]]))

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
        reverse = True
        for epoch in range(self.epochs):
            reverse = not reverse
            with open(path, "r") as file:
                for line in file.readlines():
                    X, y = self.serialize(line, reverse=reverse)
                    o = self.forward(X)
                    self.backward(X, y, o)
            print(f"{epoch}: {numpy.array(self.loss).mean()}")

    def save_weights(self):
        """
        Save model weights
        """
        with open(f"weights/{self.name}.json", "w") as file:
            json.dump({"w1": self.w1.tolist(), "w2": self.w2.tolist()}, file)


if __name__ == "__main__":

    connect = ConnectAI("epoch_10", epochs=10)
    connect.train("connect-4.data")
    connect.save_weights()
