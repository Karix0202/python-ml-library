import numpy as np
from activations import *

class Layer(object):
    def __init__(self, n_input, n_output, eta, act=Identity, W=None, b=None):
        self.rnd_state = rnd_gen = np.random.RandomState(1)
        self.n_input = n_input
        self.n_output = n_output
        self.eta = eta
        self.act = act

        if W == None:
            self.W = self.rnd_state.normal(loc=0, scale=.01, size=(n_input, n_output))
        else:
            self.W = W

        if b == None:
            self.b = np.zeros((n_output))
        else:
            self.b = b
