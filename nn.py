from layers import *

class Model(object):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise Exception('Layer must be instance of layers.Layer class')

        self.layers.append(layer)

    def fit(self, X, y, eta=.001, verbose=True):
        pass
