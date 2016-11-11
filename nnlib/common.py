import numpy as np

# Simple and common helper functions

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_d(x):
    sig = sigmoid(x)
    return sig * (sig - 1.0)

def tanh(x):
    return np.tanh(x)

def tanh_d(x):
    th = tanh(x)
    return 1.0 - np.power(th, 2)

def relu(x):
    return np.maximum(0.0, x)

def relu_d(x):
    return 1.0 * (x > 0)

def softmax(x, axis=1):
    # to make the softmax a "safe" operation we will 
    # first subtract the maximum along the specified axis
    # so that np.exp(x) does not blow up!
    # Note that this does not change the output.
    x_max = np.max(x, axis=axis, keepdims=True)
    x_safe = x - x_max
    e_x = np.exp(x_safe)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels

def unhot(one_hot_labels):
    """ Invert a one hot encoding, creating a flat vector """
    return np.argmax(one_hot_labels, axis=-1)

