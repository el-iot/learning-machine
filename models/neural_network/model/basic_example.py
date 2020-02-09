from .. import NeuralNetwork

if __name__ == "__main__":

    nn = NeuralNetwork([3, 2, 1], has_bias=True, loss_function=abs, learning_rate=0.000005)
    nn.train('data/path')

