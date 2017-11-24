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
        #return (probs == probs.max(axis=1, keepdims=1)).astype(int)
        return probs

    def test_Network(self, testing_data):
        total_error = 0
        test_results = 0
        mnist = MNIST_load_data()
        for (x,y) in testing_data:
            probs = self.feed_Network(x)
            logprob = -np.log(probs)
            total_error += np.dot(logprob,mnist.one_hot_vectors(y))
            if(np.argmax(probs) == y):
                test_results += 1
        #test_results = [(np.argmax(self.feed_Network(x)), y) for (x, y) in testing_data]
        #pdb.set_trace()
        return test_results,total_error

    def Stochastic_Gradient_Descent(self, validation_data, training_data, learning_rate):

        iterations = 3
        SGD_Size = 10
        if testing_data: n_test = len(validation_data)
        n = len(training_data)
        error = 999

        for j in xrange(iterations):
            random.shuffle(training_data)
            SGD_Batches = [
                training_data[k:k + SGD_Size]
                for k in xrange(0, n, SGD_Size)
            ]
            total_error = 0
            for SGD_Batch in SGD_Batches:
                total_error += self.update_SGD_Batch(SGD_Batch, learning_rate)

            print 'Training Error:' , total_error[0][0]/len(training_data)

            if validation_data:
                previous_error = error
                accuracy,error = self.test_Network(validation_data)
                error = np.true_divide(error,n_test)
                #pdb.set_trace()
                print "Validation: iteration {0}:  accuracy={1}% with error {2}".format(j, np.true_divide(accuracy,n_test)*100, error[0][0])
                if(previous_error < error):
                    break

        accuracy, error = self.test_Network(testing_data)
        print "Testing:  accuracy={1}% with error {2}".format(j, np.true_divide(accuracy, n_test) * 100,np.true_divide(error[0][0], n_test))

    def update_weights(self, weight, learning_rate, length, nabla):

        return weight - ((learning_rate / length) * nabla)

    def update_SGD_Batch(self, SGD_Batch, learning_rate):

        batch_bias = [np.zeros(b.shape) for b in self.biases]
        batch_weight = [np.zeros(w.shape) for w in self.weights]
        total_error = 0
        for dig, op in SGD_Batch:
            delta_bias, delta_weight, error = self.back_Propogation(dig, op)
            total_error += error
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

        return total_error

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
        logprob = -np.log(probs)
        error = np.dot(logprob, op)
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
        return (batch_bias, batch_weight,error)

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
        # training for only 10000 data set
        training_inputs = [training_inputs[i:i + 40000] for i in xrange(0, len(training_inputs), 40000)]
        training_results = [training_results[i:i + 40000] for i in xrange(0, len(training_results), 40000)]
        training_data = zip(training_inputs[0], training_results[0])
        #pdb.set_trace()
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
    network.Stochastic_Gradient_Descent(validation_data, training_data, 0.01)
    
