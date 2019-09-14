#!/Users/el/myenv/bin/python
"""
PyTorch Implementation of a basic neural network
"""
from random import randint

import torch
from pandas import read_csv

ROWS = 100  # how many rows from the basic dataset to use
EPOCHS = 1000


def sigmoid_activation(z):
    return 1 / (1 + torch.exp(-z))


def sigmoid_delta(x):
    return x * (1 - x)


class NeuralNetwork:
    """
    Neural Network Class
    """

    def __init__(self, n_input, n_hidden, n_output):

        self.w1 = torch.randn(n_input, n_hidden)  # weight for hidden layer
        self.w2 = torch.randn(n_hidden, n_output)  # weight for output layer

        self.b1 = torch.randn((1, n_hidden))  # bias for hidden layer
        self.b2 = torch.randn((1, n_output))  # bias for output layer

        self.learning_rate = 0.000001
        self.loss = []

    def feed_forward(self, X):

        self.z1 = torch.mm(X, self.w1) + self.b1
        self.a1 = sigmoid_activation(self.z1)

        self.z2 = torch.mm(self.a1, self.w2) + self.b2
        self.output = sigmoid_activation(self.z2)

    def back_propagate(self, X, y):

        loss = y - self.output

        delta_output = sigmoid_delta(self.output)
        delta_hidden = sigmoid_delta(self.a1)

        d_outp = loss * delta_output
        loss_h = torch.mm(d_outp, self.w2.t())
        d_hidn = loss_h * delta_hidden

        self.w2 += torch.mm(self.a1.t(), d_outp) * self.learning_rate
        self.w1 += torch.mm(X.t(), d_hidn) * self.learning_rate
        self.b2 += d_outp.sum() * self.learning_rate
        self.b1 += d_hidn.sum() * self.learning_rate

        self.loss += [loss]

    def train(self, X, y):

        self.feed_forward(X)
        self.back_propagate(X, y)


if __name__ == "__main__":

    network = NeuralNetwork(3, 1, 1)
    data = read_csv("../../data/basic_data.csv", index_col="Unnamed: 0").head(ROWS)
    for _ in range(EPOCHS):
        for i, row in data.iterrows():
            X = torch.tensor([row[:-1]])
            y = torch.tensor(row[-1])
            network.train(X, y)
        print(f"Loss: {sum(network.loss)/ROWS}")
        network.loss = []
