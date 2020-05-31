import numpy as np

class Cost(object):
    def __call__(self, y_hat, y_true):
        return None

    def mean(self, y_hat, y_true):
        return None

    def derivative(y_hat, y_true):
        return None
