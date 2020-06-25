import numpy as np
import matplotlib.pyplot as plt
from supervised_class.util import get_data as mnist # code from udemy tutorial
from datetime import datetime

from perceptron import Perceptron

def get_simple_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    return (X, Y)

if __name__ == '__main__':
    model = Perceptron()

    X, Y = mnist()
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]
    Y[Y == 0] = -1
    model = Perceptron()
    t0 = datetime.now()
    model.fit(X, Y, learning_rate=10e-3)
    print("MNIST train accuracy: " + str(model.score(X, Y)))

    print("\nXOR results: ")
    X, Y = get_simple_xor()
    model.fit(X, Y)
    print("XOR accuracy: " + str(model.score(X, Y)))
