""" 1. This program slightly  incorporates ideas from tutorial blog and implementation of neural
    network by Michael Nielsen(http://neuralnetworksanddeeplearning.com/chap1.html) and from boook Neural network and deep learning
    2. We also use python-mnist 0.3, standard python library authored by Richard Marko avaialable
    at https://pypi.python.org/pypi/python-mnist"""
#Assumptions:
# 1. Considering the limited processing speed of our system, our stopping criteria is either completing 30 iterations/epoch
# or attaining an accuracy of 90% without dropout and 95% with dropout(To show that using dropout higher accuracy can me met only with lower dropout)
#2. Accuracy can be higher if the stopping criterias are changed
import os
import random
# Third-party libraries
import numpy as np
from mnist import MNIST

class Neural_Network(object):

    def __init__(self):

        np.random.seed(0)
        self.biases = []
        self.weights = []
        self.P_weights = []
        self.P_biases  = []
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

    def test_Network(self, testing_data,dropout):
        total_error = 0
        test_results = 0
        mnist = MNIST_load_data()
        for (x,y) in testing_data:
            probs = self.feed_Network(x,dropout)
            logprob = -np.log(probs)
            op = mnist.one_hot_vectors(y)
            total_error += np.dot(logprob, op)
            if(np.argmax(probs) == y):
                test_results += 1
        return test_results,total_error


    def Stochastic_Gradient_Descent(self, validation_data, training_data, testing_data, learning_rate, dropout):

        iterations = 30
        SGD_Size = 10
        n_val = len(validation_data)
        n_test = len(testing_data)
        n = len(training_data)

        for j in xrange(iterations):
            m2 = np.random.binomial(1, dropout, size=(1, 256))
            m3 = np.random.binomial(1, dropout, size=(1, 256))
            random.shuffle(training_data)
            SGD_Batches = [
                training_data[k:k + SGD_Size]
                for k in xrange(0, n, SGD_Size)
            ]
            total_error = 0
            for SGD_Batch in SGD_Batches:
                total_error += self.update_SGD_Batch(SGD_Batch, learning_rate, m2,m3)
            #print 'Training Error:' , total_error[0][0]/len(training_data)

            if validation_data:

                accuracy,error = self.test_Network(validation_data,dropout)
                error = np.true_divide(error,n_val)
                accuracy = np.true_divide(accuracy,n_val)*100
                print "Validation: iteration {0}:  accuracy={1}% with Cost {2}".format(j, accuracy, error[0][0])
                #accuracy1, error1 = self.test_Network(testing_data,dropout)
                #print "Testing:  accuracy={0}% with cost {1}".format(np.true_divide(accuracy1, n_test) * 100, np.true_divide(error1[0][0], n_test))
                if(dropout < 1):
                    if (accuracy >= 92):
                            break
                else:
                    if(accuracy > 90):
                        break

        accuracy, error = self.test_Network(testing_data,dropout)
        print "Testing:  accuracy={1}% with cost {2}".format(j, np.true_divide(accuracy, n_test) * 100,np.true_divide(error[0][0], n_test))
        print "Increase the number of iterations if the accuracy obtained is below expected accuracy"

    def update_values(self, old_values, learning_rate, length, new_values):

        return old_values - ((learning_rate / length) * new_values)

    def update_SGD_Batch(self, SGD_Batch, learning_rate, m2,m3):

        batch_bias = [np.zeros(b.shape) for b in self.biases]
        batch_weight = [np.zeros(w.shape) for w in self.weights]
        total_error = 0
        for dig, op in SGD_Batch:
            delta_bias, delta_weight, error = self.back_Propogation(dig, op, m2, m3)
            #update weights
            total_error += error
            batch_weight[0] += delta_weight[0].transpose()
            batch_weight[1] += delta_weight[1].transpose()
            batch_weight[2] += delta_weight[2]
            batch_bias[0] += delta_bias[0]
            batch_bias[1] += delta_bias[1]
            batch_bias[2] += delta_bias[2]

        self.weights = list(
            map(
                lambda (i, x): self.update_values(
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
                lambda (i, x): self.update_values(
                    x,
                    learning_rate,
                    len(SGD_Batch),
                    batch_bias[i]
                ),
                enumerate(self.biases)
            )
        )
        return total_error

    def back_Propogation(self, dig, op, m2, m3):

        batch_bias = [np.zeros(b.shape) for b in self.biases]
        batch_weight = [np.zeros(w.shape) for w in self.weights]

        #feed forward
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
        logprob = -np.log(probs)
        error = np.dot(logprob, op)
        activations.append(probs)

        # back propogation

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
        # training for only 50000 data set
        training_inputs = [training_inputs[i:i + 50000] for i in xrange(0, len(training_inputs), 50000)]
        training_results = [training_results[i:i + 50000] for i in xrange(0, len(training_results), 50000)]
        training_data = zip(training_inputs[0], training_results[0])
        test_inputs = [np.reshape(x, (784, 1)) for x in test[0]]
        # Dividing dataset into testing and validation of 5000 each
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
    dropout = 0.5 #probabity of dropping a neuron(0 is no dropout, 1 is dropping all neurons)
    learning_rate = 1.5
    training_data, validation_data, testing_data = mnist.load_data_wrapper()
    network = Neural_Network()
    network.Stochastic_Gradient_Descent(validation_data, training_data, testing_data, learning_rate, 1 - dropout)
