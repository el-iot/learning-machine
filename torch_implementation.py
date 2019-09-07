#!/Users/el/myenv/bin/python
"""
PyTorch Implementation of a basic neural network
"""
import torch
from random import randint


def sigmoid_activation(z):
    return 1 / (1 + torch.exp(-z))


def sigmoid_delta(x):
    return x * (1 - x)


class NeuralNetwork:
    """
    Neural Network Class
    """

    def __init__(self, n_input, n_hidden, n_output):

        # initialize tensor variables for weights
        self.w1 = torch.randn(n_input, n_hidden)  # weight for hidden layer
        self.w2 = torch.randn(n_hidden, n_output)  # weight for output layer

        # initialize tensor variables for bias terms
        self.b1 = torch.randn((1, n_hidden))  # bias for hidden layer
        self.b2 = torch.randn((1, n_output))  # bias for output layer

        self.learning_rate = 0.001

    def feed_forward(self, X):

        self.z1 = torch.mm(X, self.w1) + self.b1
        self.a1 = sigmoid_activation(self.z1)

        # activation (output) of final layer
        self.z2 = torch.mm(self.a1, self.w2) + self.b2
        self.output = sigmoid_activation(self.z2)

    def back_propagate(self, X, y):

        loss = y - self.output
        print(loss)

        delta_output = sigmoid_delta(self.output)
        delta_hidden = sigmoid_delta(self.a1)

        # backpass the changes to previous layers
        d_outp = loss * delta_output
        loss_h = torch.mm(d_outp, self.w2.t())
        d_hidn = loss_h * delta_hidden

        self.w2 += torch.mm(self.a1.t(), d_outp) * self.learning_rate
        self.w1 += torch.mm(X.t(), d_hidn) * self.learning_rate
        self.b2 += d_outp.sum() * self.learning_rate
        self.b1 += d_hidn.sum() * self.learning_rate

    def train(self, X, y):

        self.feed_forward(X)
        self.back_propagate(X, y)


if __name__ == "__main__":

    value = torch.rand(1, 1)
    network = NeuralNetwork(1, 2, 1)
    for i in range(1000000):
        network.train(value, value)
