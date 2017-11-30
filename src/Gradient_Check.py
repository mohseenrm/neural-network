""" This program slightly incorporates ideas from tutorial blog and implementation of neural
    network by Ottokar Tilk(https://gist.github.com/ottokart/ebd3d32438c13a62ea3c)"""

import os
import random
import pdb
# Third-party libraries
import numpy as np
from mnist import MNIST

class Neural_Network(object):

    def __init__(self):

        np.random.seed(0)
        self.biases = []
        self.weights = []
        self.weights.append(np.random.randn(784, 256))
        self.biases.append(np.random.randn(1, 256))
        self.weights.append(np.random.randn(256, 256))
        self.biases.append(np.random.randn(1, 256))
        self.weights.append(np.random.randn(256, 10))
        self.biases.append(np.random.randn(1, 10))

    def feed_Network(self, a,dropout):
        scaled_a = np.true_divide(a, 255)
        z1 = (scaled_a.transpose()).dot(self.weights[0]) + self.biases[0]
        a1 = sigmoid(z1) * dropout
        z2 = (a1.dot(self.weights[1]) + self.biases[1])
        a2 = sigmoid(z2) * dropout
        z3 = (a2.dot(self.weights[2]) + self.biases[2])
        exp_scores = np.exp(z3)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs


    def back_Propogation(self, dig, op, m2, m3):


        batch_bias = [np.zeros(b.shape) for b in self.biases]
        batch_weight = [np.zeros(w.shape) for w in self.weights]

        scaled_x = np.true_divide(dig,255)
        activation_fn = scaled_x
        activations = [scaled_x]
        activation_layer = []

        activation= np.dot(activation_fn.transpose(),self.weights[0]) + self.biases[0]
        activation_layer.append(activation)

        activation_fn = sigmoid(activation) * m2
        activations.append(activation_fn)

        activation = np.dot(activation_fn,self.weights[1]) + self.biases[1]
        activation_layer.append(activation)
        activation_fn = sigmoid(activation) *m3
        activations.append(activation_fn)

        activation = np.dot(activation_fn,self.weights[2])+self.biases[2]
        activation_layer.append(activation)
        exp_scores = np.exp(activation)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        activations.append(probs)

        # backward pass

        gradient = activations[3] - op.transpose()
        batch_weight[2] = (activations[2].transpose()).dot(gradient)
        batch_bias[2] = np.sum(gradient, axis=0, keepdims=True)

        sigmoid_prime = sigmoid(activation_layer[-2],True)
        gradient = np.dot(self.weights[2], gradient.transpose()).transpose() * sigmoid_prime * m3
        batch_bias[1] = gradient
        batch_weight[1] = np.dot(gradient.transpose(), activations[1])

        sigmoid_prime = sigmoid(activation_layer[-3],True)
        gradient = np.dot(self.weights[1], gradient.transpose()).transpose() * sigmoid_prime * m2
        batch_bias[0] = gradient
        batch_weight[0] = np.dot(gradient.transpose(), activations[0].transpose())


        return (batch_weight)

    def check_gradients(self,inp,op, dropout):

        tiny = 1e-4
        m2 = np.random.binomial(1, dropout, size=(1, 256))
        m3 = np.random.binomial(1, dropout, size=(1, 256))
        W1 = [256,256,256]
        W2 = [784,256,10]
        count = 0
        for i in range(3):
            #print 'I {}'.format(i)
            for j in range(W1[i]):
                #print 'J {}'.format(j)
                for k in range(W2[i]):
                    #print 'K {}'.format(k)
                    np.random.seed(1)
                    #pdb.set_trace()

                    dW = self.back_Propogation(inp, op, m2, m3)
                    gradient1 = dW[i][j,k]

                    np.random.seed(1)
                    if(i==0):
                        self.weights[i][k,j] -= tiny
                    else:
                        self.weights[i][j,k] -= tiny

                    probs = self.feed_Network(inp, dropout)
                    logprob = -np.log(probs)
                    error1 = np.dot(logprob, op)

                    np.random.seed(1)
                    if (i == 0):
                        self.weights[i][k, j] += 2 * tiny
                    else:
                        self.weights[i][j, k] += 2 * tiny

                    #self.weights[i][j,k] += 2 * tiny
                    probs = self.feed_Network(inp, dropout)
                    logprob = -np.log(probs)
                    error2 = np.dot(logprob, op)

                    if (i == 0):
                        self.weights[i][k, j]  -= tiny
                    else:
                        self.weights[i][j, k]  -= tiny

                    #self.weights[i][j,k] -= tiny

                    gradient2 = (error2 - error1) / (2 * tiny)

                    if(gradient1-gradient2[0][0] > 1e-5):
                        count = count + 1
                        #print 'incorrect'
                        print gradient1-gradient2[0][0]

        print "Gradients OK"
        print "Incorrect:{}/{}".format(count,268800)


def sigmoid(z,derivative = False):
    if derivative:
        return sigmoid(z) * (1 - sigmoid(z))
    else:
        return 1.0/(1.0+np.exp(-z))

class MNIST_load_data(object):
    def __init__(self):
        pass

    def load_data_wrapper(self):

        data_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '..',
                'data'
            )
        )
        mndata = MNIST(data_path)
        test = mndata.load_testing()

        test_inputs = [np.reshape(x, (784, 1)) for x in test[0]]
        inputs = [test_inputs[i:i + 5000] for i in xrange(0, len(test_inputs),5000)]
        labels = [test[1][i:i + 5000] for i in xrange(0, len(test[1]), 5000)]
        testing_data = zip(inputs[0], labels[0])
        #validation_data = zip(inputs[1], labels[1])

        return (testing_data)


    def one_hot_vectors(self, j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

if __name__ == "__main__":

    mnist = MNIST_load_data()
    dropout = 1 #probabity of retaining a neuron(1 is no dropout, 0 is dropping all neurons)
    learning_rate = 0.01
    testing_data = mnist.load_data_wrapper()
    network = Neural_Network()
    for (x, y) in testing_data:
        network.check_gradients(x,mnist.one_hot_vectors(y),dropout)
        break
