import numpy

__all__ = ["sigmoid", "sigmoid_prime", "htan", "htan_prime"]


def sigmoid(s):
    """
    Sigmoid Activation Function
    """
    return 1 / (1 + numpy.exp(-s))


def sigmoid_prime(s):
    """
    Derivative of the sigmoid activation function
    """
    return s * (1 - s)


def htan(s):
    """
    Hyperbolic Tan Activation Function
    """
    return 2 / (1 + numpy.exp(-2 * s)) - 1


def htan_prime(s):
    """
    Derivative of the hyperbolic tan activation function
    """
    return 1 - s ** 2


def leaky_relu(s):
    """
    Leaky RELU activation function
    """
    return numpy.where(s > 0, s, 0.1 * s)


def leaky_relu_prime(s):
    """
    Derivative of the leaky RELU activation function
    """
    return numpy.where(s > 0, 1, 0.1)


def linear(s):

    """
    Linear Activation Function
    """
    return s


def linear_prime(s):
    """
    Derivative of the linear activation function
    """
    return 1


def elu(s):
    """
    ELU activation function
    """
    return numpy.where(s > 0, s, 0.01 * (numpy.exp(s) - 1))


def elu_prime(s):
    """
    Derivative of the ELU activation function
    """
    return numpy.where(s > 0, 1, 0.01 * numpy.exp(s))
