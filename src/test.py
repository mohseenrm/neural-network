import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import cPickle
import gzip

X, y = sklearn.datasets.make_moons(200, noise=0.20)

def load_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(training_inputs, training_results) #X,y
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# %% 15 
num_examples = len(X)  # training set size
nn_input_dim = 784  # input layer dimensionality
nn_output_dim = 10  # output layer dimensionality
nn_hdim = 286 #hidden layer dimentionality

# Gradient descent parameters (I picked these by hand) 
epsilon = 0.01  # learning rate for gradient descent
reg_lambda = 0.01  # regularization strength

dropout = 0.5 # 1.0 = no dropout
training=False


# %% 7
# Helper function to evaluate the total loss on the dataset 
def calculate_loss(model):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model ['b3']
    # Forward propagation to calculate our predictions 
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss 
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional) 
    # data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


# %% 8
# Helper function to predict an output (0 or 1) 
def predict(model, x):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation 
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer 
# - num_passes: Number of passes through the training data for gradient descent 
# - print_loss: If True, print the loss every 1000 iterations 
def build_model(num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_hdim))
    W3 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b3 = np.zeros((1, nn_output_dim))

    # This is what we return at the end 
    model = {}

    # Gradient descent. For each batch... 
    for i in range(0, num_passes):

        # Forward propagation 
        z1 = X.dot(W1) + b1 #np.dot(w, activation) + b
        a1 = sigmoid(z1)
        # Dropout in layer 1
        if training:
            m2 = np.random.binomial(1, dropout, size=z1.shape)
        else:
            m2 = dropout #prob
        a1 *= m2

        z2 = a1.dot(W2) + b2
        a2 = sigmoid(z2)


        # Dropout in layer 2
        if training:
            m2 = np.random.binomial(1, dropout, size=z2.shape)
        else:
            m2 = dropout  # prob
        a2 *= m2

        z3 = a2.dot(W3) + b3
        exp_scores = np.exp(z3)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation 
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW3 = (a2.T).dot(delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)

        #sigmoid
        sp1 = sigmoid(z1)
        sp2 = sigmoid(z2)
        delta2 = np.dot(W3.transpose(), delta3) * sp2
        dW2 = np.dot(delta2, a2.transpose())
        db2 = delta2

        delta1 = np.dot(W3.transpose(), delta2) * sp1
        dW1 = np.dot(delta2, a1.transpose())
        db1 = delta1


        # Add regularization terms (b1 and b2 don't have regularization terms) 
        dW3 += reg_lambda * W3
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update 
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        W3 += -epsilon * dW3
        b3 += -epsilon * db3

        # Assign new parameters to the model 
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 , 'W3': W3, 'b3': b3}

        # Optionally print the loss. 
        # This is expensive because it uses the whole dataset, so we don't want to do it too often. 
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model)))

    return model

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


# Build a model with a 3-dimensional hidden layer
model = build_model(3, print_loss=True)