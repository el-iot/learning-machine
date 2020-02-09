#!/Users/el/myenv/bin/python
"""
Numpy Implementation of a neural network
"""

import json
from typing import List

import numpy

from model.neural_network import NeuralNetwork


class ConnectThought(NeuralNetwork):
    """
    Connect-Four AI
    """

    def __init__(self, shape, name="connect-thought"):
        """
        Initialise the model
        """

        super().__init__(shape, name=name)

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
        X = self.flatten(
            [self.convert(x) for x in instance.split(",")][:-1][:: (-1 if reverse else 1)]
        )
        y = self.convert(instance.split(",")[-1])
        return (numpy.array([X + [1]]), numpy.array([y]))  # bias neuron

    def flatten(self, x):
        """
        Flattens a list of nested lists
        """
        if not isinstance(x, list):
            return [x]
        return [w for v in x for w in self.flatten(v)]

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
        raise NotImplementedError()


if __name__ == "__main__":

    connect = ConnectThought([85, 6, 3, 1])
    connect.train("connect-4.data")
