import numpy as np


def pass_through(x):
    id = 0
    return x


def logistic(x):
    return 1 / (1 + np.exp(-x))


def leaky_ReLU(x):
    if x > 0:
        return x
    else:
        return x * .01


def tan_h(x):
    return np.tanh(x)
