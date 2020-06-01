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

            grad = y
            for i in range(len(self.layers))[::-1]:
                grad = self.layers[i].backward(grad)

            if verbose:
                print('ITER: {}, ERR: {}'.format(iter+1, cost().mean(self.predict(X), y)))

    def predict(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)

        return out

    def pass_args(self, args):
        self.layers[-1].cost = args['cost']()

        for layer in self.layers:
            layer.eta = args['eta']

    def check_layers(self):
        if len(self.layers) <= 0:
            raise Exception('Model must have at least one layer')

        for i in range(len(self.layers)):
            if i != len(self.layers) - 1 and len(self.layers) == 1:
                if isinstance(self.layers[i], OutputLayer):
                    raise Exception('There can be only one OutputLayer')

        if not isinstance(self.layers[-1], OutputLayer):
            raise Exception('Last layer must be instance of layers.OutputLayer class')
