
The code is written in Python and requires a python 2.7 compiler to execute.
Libraries Used:
Standart Python Library: OS, Random, Numpy
External Library: python-mnist 0.3( Installation using PIP: pip install python-mnist )

Data Set: http://yann.lecun.com/exdb/mnist/
Download the datasets for training and testing i.e. train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, 
t10k-images-idx3-ubyte.gz
Unzip Them to get 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte'
Copy the unzipped files without renaming them to src/data. (Have to be the same name as mentioned above)

To train and test the neural network:
Move to src, then type python NeuralNetwork.py
It trains using training data and outputs the accuracy using validation and testing data.
To change parameter:
dropout: change value of dropout in main #default: 0
learning rate: change the value of learning_rate in main #default:1.5
iterations or epoch: change the value of iteration in Stochastic_Gradient_Descent #default: 30
SGD batch size (Epoch size): change the value of SGD_Size in Stochastic_Gradient_Descent #default: 10

To test the gradients :
Move to src, then type python gradient_Check.py
It uses default testing data and outputs the gradients that have a difference greater than 1e-4.
To change parameter:
dropout: change value of dropout in main #default: 0
learning rate: change the value of learning_rate in main #default:1.5