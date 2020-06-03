from layers import *
from costs import *

class Model(object):
    def __init__(self):
        self.layers = []
        self.training_errors = []

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise Exception('Layer must be instance of layers.Layer class')

        self.layers.append(layer)

    def fit(self, X, y, cost, n_iter=1000, eta=.001, verbose=True):
        self.check_layers()
        self.pass_args({
            'eta': eta,
            'cost': cost,
        })

        for iter in range(n_iter):
            out = X
            for layer in self.layers:
                out = layer.forward(out)

            self.backward(out, y, cost)
            err = cost().mean(self.predict(X), y)
            self.training_errors.append(err)

            if verbose and iter % 10 == 9:
                print('ITER: {}, ERR: {}'.format(iter+1, err))

    def predict(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)

        return out

    def pass_args(self, args):
        for layer in self.layers:
            layer.eta = args['eta']

    def check_layers(self):
        if len(self.layers) <= 0:
            raise Exception('Model must have at least one layer')

    def backward(self, output, y_true, cost):
        grad = cost().derivative(output, y_true)

        for i in range(len(self.layers))[::-1]:
            grad = self.layers[-1].backward(grad)
