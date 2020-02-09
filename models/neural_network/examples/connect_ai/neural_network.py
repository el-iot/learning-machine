#!/Users/el/myenv/bin/python
"""
Numpy Implementation of a neural network
"""

import json
from typing import List

import numpy


class ConnectAI(NeuralNetwork):
    """
    Connect-Four AI
    Uses a sigmoid activation function
    """

    def __init__(self, shape, name, epochs=20):
        """
        Initialise the model
        """

        super().__init__(shape, name=name)
        # shape 85, 6, 3, 1

    def convert(self, entry):
        """
        Convert an entry to a binary representation
        """
        return {"x": [1, 0], "o": [0, 1], "b": [0, 0], "win": 1, "loss": -1, "draw": 0}[
            entry.replace("\n", "")
        ]

    def ternary_to_binary(self, ternary: List, length=67):
        """
        Converts a ternary list-representation of a number
        into the binary list-representation of that number
        """
        if isinstance(ternary, list):
            ternary = "".join([str(_) for _ in ternary])
        binary = bin(
            sum([3 ** (len(ternary) - i - 1) * int(ternary[i]) for i in range(len(ternary))])
        )[2:]
        return [int(x) for x in "0" * (length - len(binary)) + binary]

    def serialize(self, instance, reverse=False):
        """
        Serialize a line into 0's and 1's
        """
        X = self.flatten([convert(x) for x in instance.split(",")][:-1][:: (-1 if reverse else 1)])
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
