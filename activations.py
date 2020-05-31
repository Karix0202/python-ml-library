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
