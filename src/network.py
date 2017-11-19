import cPickle as pickle
import gzip
import os
import random
import pdb
# Third-party libraries
import numpy as np
from mnist import MNIST

class Network(object):
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

    def feedforward(self, a):
        scaled_a = np.divide(a,1)
        z1 = (scaled_a.transpose()).dot(self.weights[0]) + self.biases[0]
        m2 = np.random.binomial(1, 1, size=z1.shape)
        a1 = sigmoid(z1)*m2
        z2 = a1.dot(self.weights[1]) + self.biases[1]
        m3 = np.random.binomial(1, 1, size=z2.shape)
        a2 = sigmoid(z2)*m3
        z3 = a2.dot(self.weights[2]) + self.biases[2]
        exp_scores = np.exp(z3)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return (probs == probs.max(axis=1, keepdims=1)).astype(int)

    def SGD(self, training_data, epochs, mini_batch_size, eta):

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_weights(self, weight, eta, length, nabla):
        return weight - ((eta / length) * nabla)

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_w[0] += delta_nabla_w[0].transpose()
            nabla_w[1] += delta_nabla_w[1].transpose()
            nabla_w[2] += delta_nabla_w[2]
            nabla_b[0] += delta_nabla_b[0]
            nabla_b[1] += delta_nabla_b[1]
            nabla_b[2] += delta_nabla_b[2]

        # Functional approach
        self.weights = list(
            map(
                lambda (i, x): self.update_weights(
                    x,
                    eta,
                    len(mini_batch),
                    nabla_w[i]
                ),
                enumerate(self.weights)
            )
        )
        self.biases = list(
            map(
                lambda (i, x): self.update_weights(
                    x,
                    eta,
                    len(mini_batch),
                    nabla_b[i]
                ),
                enumerate(self.biases)
            )
        )

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        scaled_x = np.divide(x,1)
        activation = scaled_x
        activations = [scaled_x]
        zs = []

        z = np.dot(activation.transpose(),self.weights[0]) + self.biases[0]
        zs.append(z)

        activation = sigmoid(z)
        activations.append(activation)

        z = np.dot(activation,self.weights[1]) + self.biases[1]
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

        z = np.dot(activation,self.weights[2])+self.biases[2]
        zs.append(z)
        exp_scores = np.exp(z)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        activations.append(probs)

        # backward pass

        delta = activations[3] - y.transpose()
        nabla_w[2] = (activations[2].transpose()).dot(delta)
        nabla_b[2] = np.sum(delta, axis=0, keepdims=True)

        sp = sigmoid(zs[-2],True)
        delta = np.dot(self.weights[2], delta.transpose()).transpose() * sp
        nabla_b[1] = delta
        nabla_w[1] = np.dot(delta, activations[1].transpose())

        sp = sigmoid(zs[-3],True)
        delta = np.dot(self.weights[1], delta.transpose()).transpose() * sp
        nabla_b[0] = delta
        nabla_w[0] = np.dot(delta.transpose(), activations[0].transpose())
        #pdb.set_trace()
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z,derivative = False):
    if derivative:
        return sigmoid(z) * (1 - sigmoid(z))
    else:
        return 1.0/(1.0+np.exp(-z))

class MNIST1(object):
    def __init__(self):
        pass

    def load_data_wrapper(self):

        #change path
        mndata = MNIST('C:\Users\Nagarchith Balaji\Desktop\Fall-17\FSL\Project\\naga\\neural-network\data')
        train = mndata.load_training()
        test = mndata.load_testing()
        #pdb.set_trace()
        """" pdb.set_trace()

               file_path = os.path.abspath(
                   os.path.join(
                       os.path.dirname(__file__),
                       '..',
                       'data',
                       'mnist.pkl.gz'
                   )
               )

               f = gzip.open(file_path, 'rb')
               tr_d, va_d, te_d = pickle.load(f)
               f.close()"""

        training_inputs = [np.reshape(x, (784, 1)) for x in train[0]]
        training_results = [self.vectorized_result(y) for y in train[1]]
        training_data = zip(training_inputs, training_results)
        validation_inputs = [np.reshape(x, (784, 1)) for x in test[0]]
        validation_data = zip(validation_inputs, test[1])
        test_inputs = [np.reshape(x, (784, 1)) for x in test[0]]
        test_data = zip(test_inputs, test[1])

        return (training_data, validation_data, test_data)




    def vectorized_result(self, j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

if __name__ == "__main__":
    mnist = MNIST1()
    training_data, validation_data, test_data = mnist.load_data_wrapper()
    net = Network()
    net.SGD(training_data, 3, 10, 0.01)
    