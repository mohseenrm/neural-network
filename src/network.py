import cPickle
import gzip
import os
import random
# Third-party libraries
import numpy as np

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

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
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
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples (x, y), and "eta"
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
        activation = x
        activations = [x]
        zs = []

        z = np.dot(activation.transpose(),self.weights[0]) + self.biases[0]
        m2 = np.random.binomial(1, 0.5, size=z.shape)
        zs.append(z)

        activation = sigmoid(z) * m2
        activations.append(activation)

        z = np.dot(activation,self.weights[1]) + self.biases[1]
        m3 = np.random.binomial(1, 0.5, size=z.shape)
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
        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class MNIST(object):
    def __init__(self):
        pass
    
    def load_data(self):
        """
        Return the MNIST data as a tuple containing the training data,
        the validation data, and the test data.

        The ``training_data`` is returned as a tuple with two entries.
        The first entry contains the actual training images.  This is a
        numpy ndarray with 50,000 entries.  Each entry is, in turn, a
        numpy ndarray with 784 values, representing the 28 * 28 = 784
        pixels in a single MNIST image.

        The second entry in the ``training_data`` tuple is a numpy ndarray
        containing 50,000 entries.  Those entries are just the digit
        values (0...9) for the corresponding images contained in the first
        entry of the tuple.

        The ``validation_data`` and ``test_data`` are similar, except
        each contains only 10,000 images.

        This is a nice data format, but for use in neural networks it's
        helpful to modify the format of the ``training_data`` a little.
        That's done in the wrapper function ``load_data_wrapper()``, see
        below.
        """
        file_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '..',
                'data',
                'mnist.pkl.gz'
            )
        )

        f = gzip.open(file_path, 'rb')
        training_data, validation_data, test_data = cPickle.load(f)
        f.close()
        return (training_data, validation_data, test_data)

    def load_data_wrapper(self):
        """
        Return a tuple containing ``(training_data, validation_data,
        test_data)``. Based on ``load_data``, but the format is more
        convenient for use in our implementation of neural networks.

        In particular, ``training_data`` is a list containing 50,000
        2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
        containing the input image.  ``y`` is a 10-dimensional
        numpy.ndarray representing the unit vector corresponding to the
        correct digit for ``x``.

        ``validation_data`` and ``test_data`` are lists containing 10,000
        2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
        numpy.ndarry containing the input image, and ``y`` is the
        corresponding classification, i.e., the digit values (integers)
        corresponding to ``x``.

        Obviously, this means we're using slightly different formats for
        the training data and the validation / test data.  These formats
        turn out to be the most convenient for use in our neural network
        code.
        """
        tr_d, va_d, te_d = self.load_data()
        training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        training_results = [self.vectorized_result(y) for y in tr_d[1]]
        training_data = zip(training_inputs, training_results)
        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = zip(validation_inputs, va_d[1])
        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, te_d[1])
        return (training_data, validation_data, test_data)

    def vectorized_result(self, j):
        """
        Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network.
        """
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

if __name__ == "__main__":
    mnist = MNIST()
    training_data, validation_data, test_data = mnist.load_data_wrapper()
    net = Network()
    net.SGD(training_data, 3, 10, 0.01, test_data=test_data)
    