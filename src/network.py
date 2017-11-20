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

    def feed_Network(self, a):
        scaled_a = np.true_divide(a, 255)
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

    def test_Network(self, testing_data):
        test_results = [(np.argmax(self.feed_Network(x)), y) for (x, y) in testing_data]
        return sum(int(x == y) for (x, y) in test_results)

    def Stochastic_Gradient_Descent(self, training_data, learning_rate):

        iterations = 3
        SGD_Size = 10
        if testing_data: n_test = len(testing_data)
        n = len(training_data)
        for j in xrange(iterations):
            random.shuffle(training_data)
            SGD_Batches = [
                training_data[k:k + SGD_Size]
                for k in xrange(0, n, SGD_Size)
            ]
            for SGD_Batch in SGD_Batches:
                self.update_SGD_Batch(SGD_Batch, learning_rate)
            if testing_data:
                print "iteration {0}:  accuracy={1}%".format(j, np.true_divide(self.test_Network(testing_data),n_test)*100)
            else:
                print "iteration {0} complete".format(j)

    def update_weights(self, weight, learning_rate, length, nabla):
        return weight - ((learning_rate / length) * nabla)

    def update_SGD_Batch(self, SGD_Batch, learning_rate):

        batch_bias = [np.zeros(b.shape) for b in self.biases]
        batch_weight = [np.zeros(w.shape) for w in self.weights]
        
        for dig, op in SGD_Batch:
            delta_bias, delta_weight = self.back_Propogation(dig, op)
            batch_weight[0] += delta_weight[0].transpose()
            batch_weight[1] += delta_weight[1].transpose()
            batch_weight[2] += delta_weight[2]
            batch_bias[0] += delta_bias[0]
            batch_bias[1] += delta_bias[1]
            batch_bias[2] += delta_bias[2]

        # Functional approach
        self.weights = list(
            map(
                lambda (i, x): self.update_weights(
                    x,
                    learning_rate,
                    len(SGD_Batch),
                    batch_weight[i]
                ),
                enumerate(self.weights)
            )
        )
        self.biases = list(
            map(
                lambda (i, x): self.update_weights(
                    x,
                    learning_rate,
                    len(SGD_Batch),
                    batch_bias[i]
                ),
                enumerate(self.biases)
            )
        )

    def back_Propogation(self, dig, op):

        batch_bias = [np.zeros(b.shape) for b in self.biases]
        batch_weight = [np.zeros(w.shape) for w in self.weights]

        scaled_x = np.true_divide(dig,255)
        activation_fn = scaled_x
        activations = [scaled_x]
        activation_layer = []

        activation= np.dot(activation_fn.transpose(),self.weights[0]) + self.biases[0]
        activation_layer.append(activation)

        activation_fn = sigmoid(activation)
        activations.append(activation_fn)

        activation = np.dot(activation_fn,self.weights[1]) + self.biases[1]
        activation_layer.append(activation)
        activation_fn = sigmoid(activation)
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
        gradient = np.dot(self.weights[2], gradient.transpose()).transpose() * sigmoid_prime
        batch_bias[1] = gradient
        batch_weight[1] = np.dot(gradient, activations[1].transpose())

        sigmoid_prime = sigmoid(activation_layer[-3],True)
        gradient = np.dot(self.weights[1], gradient.transpose()).transpose() * sigmoid_prime
        batch_bias[0] = gradient
        batch_weight[0] = np.dot(gradient.transpose(), activations[0].transpose())
        return (batch_bias, batch_weight)

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
        train = mndata.load_training()
        test = mndata.load_testing()

        training_inputs = [np.reshape(x, (784, 1)) for x in train[0]]
        training_results = [self.one_hot_vectors(y) for y in train[1]]
        training_data = zip(training_inputs, training_results)
        test_inputs = [np.reshape(x, (784, 1)) for x in test[0]]
        inputs = [test_inputs[i:i + 5000] for i in xrange(0, len(test_inputs),5000)]
        labels = [test[1][i:i + 5000] for i in xrange(0, len(test[1]), 5000)]
        testing_data = zip(inputs[0], labels[0])
        validation_data = zip(inputs[1], labels[1])

        return (training_data, validation_data, testing_data)


    def one_hot_vectors(self, j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

if __name__ == "__main__":
    mnist = MNIST_load_data()
    training_data, validation_data, testing_data = mnist.load_data_wrapper()
    network = Neural_Network()
    network.Stochastic_Gradient_Descent(training_data, 0.01)
    
