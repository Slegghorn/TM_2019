import math
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 1], [1, 4], [1, 2], [1, 5], [2, 1], [2, 4], [2, 2], [2, 5], [4, 4], [4, 1], [4, 5], [4, 2], [5, 4], [5, 1], [5, 5], [5, 2]]).reshape(16,2)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ]).reshape(16,1)

mean = np.mean(x)
std = np.std(x)
x = (x - mean)/std
lr = 0.03
epochs = 100


def Relu(x):
    return x * (x > 0)
def dRelu(x):
    return 1 * (x > 0)

def sig(x):
    return 1 / (1 + np.exp(-x))
def dsig(x):
    return np.exp(-x) / ((1 + np.exp(-x))**2)

def MSE(y_p, y):
    return (y - y_p)**2

class NeuralNetwork():

    def __init__(self):
        self.w1 = np.random.normal(0, 1.0, size = (2, 4))
        self.b1 = np.random.normal(size = (4))

        self.w2 = np.random.normal(0, 1.0, size = (4, 1))
        self.b2 = np.random.normal(size = (1))

    def feedforward(self, x):
        h = Relu(np.dot(x, self.w1) + self.b1)
        print(h)
        return sig(np.dot(h, self.w2) + self.b2)

    def train(self, x_train, y_train):
        for i in range(epochs):
            loss = 0
            acc = []
            for x, y in zip(x_train, y_train):
                l_1 = np.dot(x, self.w1) + self.b1
                h = Relu(l_1)
                l_2 = np.dot(h, self.w2) + self.b2
                o = sig(l_2)

                for i in range(4):
                    delta = dRelu(l_1[i]) *self.w2[i] * dsig(l_2) * -2*(y - o)
                    self.b1[i] -= lr * delta
                    for j in range(2):
                        self.w1[j][i] -= x[j] * lr * delta

                for i in range(1):
                    delta = dsig(l_2) * -2*(y - o)
                    self.b2[i] -= lr * delta
                    for j in range(4):
                        self.w2[j][i] -= h[j] * lr * delta

                loss += (y - o)**2
                a = 1 * ((1 * (o >= 0.5)) == y)
                acc.append(a[0])
            print(loss / len(x_train))
            print(sum(acc) * 100/len(x_train), '%')




nn = NeuralNetwork()
nn.train(x, y)
