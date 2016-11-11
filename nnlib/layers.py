from .common import *

from .core.layer import Layer
from .core.optim import *

import numpy as np

class Activation(object):

    def __init__(self, tname):
        if tname == 'sigmoid':
            self.act = sigmoid
            self.act_d = sigmoid_d
        elif tname == 'tanh':
            self.act = tanh
            self.act_d = tanh_d
        elif tname == 'relu':
            self.act = relu
            self.act_d = relu_d
        else:
            raise ValueError('Invalid activation function.')

    def fprop(self, input):
        # we need to remember the last input
        # so that we can calculate the derivative with respect
        # to it later on
        self.last_input = input
        return self.act(input)

    def bprop(self, output_grad):
        return output_grad * self.act_d(self.last_input)


# define a container for providing input to the network
class InputLayer(Layer):
    def __init__(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError("InputLayer requires input_shape as a tuple")
        self.input_shape = input_shape

    def output_size(self):
        return self.input_shape
    
    def fprop(self, input):
        return input
    
    def bprop(self, output_grad):
        return output_grad

# Define the 
class FullyConnectedLayer(Layer, Parameterized):
    """ A standard fully connected hidden layer
    """
    def __init__(self,
            input_layer,
            num_units,
            init_stddev,
            activation_fun=Activation('relu')):
        # Layer properties
        self.num_units = num_units
        self.activation_fun = activation_fun
        # the input shape will be of size (batch_size, num_units_prev) 
        # where num_units_prev is the number of units in the input 
        # (previous) layer
        self.input_shape = input_layer.output_size()
        # FIXME this is the weight matrix it should have shape: (num_units_prev, num_units)
        self.W = init_stddev * np.random.randn(self.input_shape[1], self.num_units)
        self.b = np.zeros(self.num_units)
        # create dummy variables for parameter gradients
        # no need to change these here!
        self.dW = None
        self.db = None

    def output_size(self):
        # shape: (batch_size, num_units)
        return (self.input_shape[0], self.num_units)

    def fprop(self, input):
        # cache the last_input for the bprop implementation below
        self.last_input = input
        # handle the case when no activation function is specified
        a = input @ self.W + self.b
        if self.activation_fun is not None:
            output = self.activation_fun.fprop(a)
        else:
            output = a
        return output

    def bprop(self, output_grad):
        # HINT: you may have to divide the weights by n
        #       to make gradient checking work 
        #       (since you want to divide the loss by number of inputs)
        n = output_grad.shape[0]
        # accumulate gradient wrt. the parameters first
        # we will need to store these to later update
        # the network after a few forward backward passes
        # the gradient wrt. W should be stored as self.dW
        # the gradient wrt. b should be stored as selfdb

        if self.activation_fun is not None:
            grad = self.activation_fun.bprop(output_grad)
        else:
            grad = output_grad

        self.dW = (self.last_input.T @ grad) / n
        self.db = np.mean(grad, axis=0) # average for the sample train
        # the gradient wrt. the input should be calculated here
        grad_input = grad @ self.W.T
        return grad_input

    def params(self):
        return self.W, self.b

    def grad_params(self):
        return self.dW, self.db

# finally we specify the interface for output layers 
# which are layers that also have a loss function
# we will implement two output layers:
#  a Linear, and Softmax (Logistic Regression) layer
# The difference between output layers and and normal 
# layers is that they will be called to compute the gradient
# of the loss through input_grad(). bprop will never 
# be called on them!
class LinearOutput(Layer, Loss):
    """ A simple linear output layer that  
        uses a squared loss (e.g. should be used for regression)
    """
    def __init__(self, input_layer):
        self.input_size = input_layer.output_size()
        
    def output_size(self):
        return (1,)
        
    def fprop(self, input):
        return input

    def bprop(self, output_grad):
        raise NotImplementedError(
            'LinearOutput should only be used as the last layer of a Network'
            + ' bprop() should thus never be called on it!'
        )
    
    def input_grad(self, Y, Y_pred):
        return Y - Y_pred #FIXME

    def loss(self, Y, Y_pred):
        loss = 0.5 * np.square(Y - Y_pred)
        return np.mean(np.sum(loss, axis=1))


class SoftmaxOutput(Layer, Loss):
    """ A softmax output layer that calculates 
        the negative log likelihood as loss
        and should be used for classification.
    """
    
    def __init__(self, input_layer):
        self.input_size = input_layer.output_size()
        
    def output_size(self):
        return (1,)
    
    def fprop(self, input):
        return softmax(input)
    
    def bprop(self, output_grad):
        raise NotImplementedError(
            'SoftmaxOutput should only be used as the last layer of a Network'
            + ' bprop() should thus never be called on it!'
        )
    
    def input_grad(self, Y, Y_pred):
        return Y_pred - Y

    def loss(self, Y, Y_pred):
        # to make the loss numerically stable 
        # you may want to add an epsilon in the log ;)
        eps = 1e-10
        n = Y.shape[0]
        # calculate negative log likelihood, then average over batch size
        loss = -1 * np.sum(Y * np.log(Y_pred + eps)) / n
        return loss

