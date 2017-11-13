
import random
import numpy as np
import tools
import time


class Network(object):

    def __init__(self, sizes):
        print "naga"
        time.sleep(50)
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.b1 = [np.random.randn(y, 1) for y in sizes[1:]]
        self.w1 = [np.random.randn(y, x)
                       for x, y in zip(sizes[:-1], sizes[1:])]
        self.b2 = [np.random.randn(y, 1) for y in sizes[1:]]
        self.w2 = [np.random.randn(y, x)
                   for x, y in zip(sizes[:-1], sizes[1:])]
        self.b3 = [np.random.randn(y, 1) for y in sizes[1:]]
        self.w3 = [np.random.randn(y, x)
                   for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.b1, self.w1):
            a1 = sigmoid(np.dot(w, a)+b)
        for b, w in zip(self.b2, self.w2):
            a2 = sigmoid(np.dot(w, a1) + b)
        for b, w in zip(self.b3, self.w3):
            result = softmax(self,np.dot(w, a2) + b)
        return result

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        n_b1 = [np.zeros(b.shape) for b in self.b1]
        n_w1 = [np.zeros(w.shape) for w in self.w1]
        n_b2 = [np.zeros(b.shape) for b in self.b2]
        n_w2 = [np.zeros(w.shape) for w in self.w2]
        n_b3 = [np.zeros(b.shape) for b in self.b3]
        n_w3 = [np.zeros(w.shape) for w in self.w3]
        for x, y in mini_batch:
            delta_b1, delta_w1, delta_b2, delta_w2, delta_b3, delta_w3, = self.backprop(x, y)
            n_b1 = [nb+dnb for nb, dnb in zip(n_b1, delta_b1)]
            n_w1 = [nw+dnw for nw, dnw in zip(n_w1, delta_w1)]
            n_b2 = [nb + dnb for nb, dnb in zip(n_b2, delta_b2)]
            n_w2 = [nw + dnw for nw, dnw in zip(n_w2, delta_w2)]
            n_b3 = [nb + dnb for nb, dnb in zip(n_b3, delta_b3)]
            n_w3 = [nw + dnw for nw, dnw in zip(n_w3, delta_w3)]
        self.w1 = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.w1, n_w1)]
        self.b1 = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.b1, n_b1)]
        self.w2 = [w - (eta / len(mini_batch)) * nw
                   for w, nw in zip(self.w2, n_w2)]
        self.b2 = [b - (eta / len(mini_batch)) * nb
                   for b, nb in zip(self.b2, n_b2)]
        self.w3 += eta * n_w3
        self.b3 += eta * n_b3

    def backprop(self, x, y):
        n_b1 = [np.zeros(b.shape) for b in self.b1]
        n_w1 = [np.zeros(w.shape) for w in self.w1]
        n_b2 = [np.zeros(b.shape) for b in self.b2]
        n_w2 = [np.zeros(w.shape) for w in self.w2]
        n_b3 = [np.zeros(b.shape) for b in self.b3]
        n_w3 = [np.zeros(w.shape) for w in self.w3]

        # feedforward

        for b, w in zip(self.b1, self.w1):
            z1 = np.dot(w, a1)+b
            a1 = sigmoid(z1)

        for b, w in zip(self.b2, self.w2):
            z2 = np.dot(w, a1)+b
            a2 = sigmoid(z2)

        for b, w in zip(self.b3, self.w3):
            z3 = np.dot(w, a2) + b
            exp_scores = np.exp(z3)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # backward pass
        delta3 = probs
        delta3[range(784), y] -= 1
        n_w3 = (a2.T).dot(delta3)
        n_b3 = np.sum(delta3, axis=0, keepdims=True)

        sp = sigmoid_prime(z2)
        delta2 = np.dot(self.w2.transpose(), delta3) * sp
        n_b2 = delta2
        n_w2 = np.dot(delta2, a2.transpose())

        sp = sigmoid_prime(z1)
        delta2 = np.dot(self.w2.transpose(), delta3) * sp
        n_b1 = delta2
        n_w1 = np.dot(delta2, a1.transpose())


        return (n_b1, n_w1,n_b2, n_w2,n_b3, n_w3)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(self, x):
    r = np.copy(x)
    for index, col in enumerate(x):
        r[index] = tools.safe_exp(col)
        r[index] = r[index] / np.sum(r[index])
    return r
