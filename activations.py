import numpy as np

class Activation(object):
    def __call__(self, input):
        return None

    def derivative(self, input):
        return None

class Sigmoid(Activation):
    def __call__(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        return np.multiply(self(input), (1 - self(1 - input)))

class ReLU(Activation):
    def __call__(self, input):
        return np.max(x, 0)

    def derivative(self, input):
        return np.heavside(input, 0)

class Identity(Activation):
    def __call__(self, input):
        return input

    def derivative(self, input):
        return np.ones(input.shape)
