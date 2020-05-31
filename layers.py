import numpy as np
from activations import *
from costs import *

class Layer(object):
    def __init__(self, n_input, n_output, eta, act=Identity, W=None, b=None):
        self.rnd_state = rnd_gen = np.random.RandomState(1)
        self.n_input = n_input
        self.n_output = n_output
        self.eta = eta
        self.act = act()

        if W == None:
            self.W = self.rnd_state.normal(loc=0, scale=.01, size=(n_input, n_output))
        else:
            self.W = W

        if b == None:
            self.b = np.zeros((n_output))
        else:
            self.b = b

    def forward(self, input):
        self.input = input
        self.output = self.act(np.dot(self.input, self.W) + np.reshape(self.b, (1, self.b.shape[0])))
        return self.output

    def backward(self, grad_last):
        grad = np.dot(self.act.derivative(self.output), grad_last)
        dW = np.dot(self.input.T, grad)
        db = grad

        self.update(dW, db)

        return np.dot(grad, self.W.T)

    def update(self, dW, db):
        self.W -= self.eta * dW
        self.b -= self.eta * db

    def get_parameters(self):
        return {
            'W': self.W,
            'b': self.b
        }

class OutputLayer(Layer):
    def __init__(self, n_input, n_output, eta, act=Identity, cost=MSE W=None, b=None):
        super().__init__(n_input, n_output, eta, act=act, W=W, b=b)
        self.cost = cost()

    def backward(self, y_true):
        loss = self.cost.derivative(self.output, y_true)
        
