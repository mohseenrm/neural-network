import mnist_loader
import numpy as np
import pdb
import random

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
        z1 = (a.transpose()).dot(self.weights[0]) + self.biases[0]
        a1 = sigmoid(z1)
        z2 = a1.dot(self.weights[1]) + self.biases[1]
        a2 = sigmoid(z2)
        z3 = a2.dot(self.weights[2]) + self.biases[2]
        exp_scores = np.exp(z3)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return (probs == probs.max(axis=1, keepdims=1)).astype(int)

    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
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

        self.weights[0] = self.weights[0] - (eta/ len(mini_batch)) * nabla_w[0]
        self.weights[1] = self.weights[1] - (eta/ len(mini_batch)) * nabla_w[1]
        self.weights[2] = self.weights[2] - (eta/ len(mini_batch)) * nabla_w[2]

        self.biases[0] = self.biases[0] - (eta / len(mini_batch)) * nabla_b[0]
        self.biases[1] = self.biases[1] - (eta / len(mini_batch)) * nabla_b[1]
        self.biases[2] = self.biases[2] - (eta / len(mini_batch)) * nabla_b[2]

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]
        zs = []

        z = np.dot(activation.transpose(),self.weights[0]) + self.biases[0]
        m2 = np.random.binomial(1, 1, size=z.shape)
        zs.append(z)
        activation = sigmoid(z) * m2
        pdb.set_trace()
        activations.append(activation)

        z = np.dot(activation,self.weights[1]) + self.biases[1]
        m3 = np.random.binomial(1, 1, size=z.shape)
        zs.append(z)
        activation = sigmoid(z) * m3
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

        sp = sigmoid_prime(zs[-2])
        delta = np.dot(self.weights[2], delta.transpose()).transpose() * sp * m3
        nabla_b[1] = delta
        nabla_w[1] = np.dot(delta, activations[1].transpose())

        sp = sigmoid_prime(zs[-3])
        delta = np.dot(self.weights[1], delta.transpose()).transpose() * sp *m2
        nabla_b[0] = delta
        nabla_w[0] = np.dot(delta.transpose(), activations[0].transpose())
        #pdb.set_trace()
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network()
    net.SGD(training_data, 3, 10, 0.01, test_data=test_data)