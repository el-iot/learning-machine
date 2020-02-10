#!/Users/el/myenv/bin/python
"""
Numpy Implementation of a neural network
"""
from model.neural_network import NeuralNetwork


class ConnectThought(NeuralNetwork):
    """
    Connect-Four AI
    """

    def __init__(self, shape, name="connect-thought", **kwargs):
        """
        Initialise the model
        """

        super().__init__(shape, name=name, **kwargs)


if __name__ == "__main__":

    connect = ConnectThought([85, 1], learning_rate=0.005)
    connect.train("../../../data/connect-4-data.csv")
