#!/Users/el/myenv/bin/python

from model.neural_network import NeuralNetwork

if __name__ == "__main__":
    nn = NeuralNetwork(
        [4, 2, 1], has_bias=True, loss_function=abs, learning_rate=0.00000005
    )
    nn.train("../../../data/basic_data.csv")
