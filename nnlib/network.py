from .common import *
from .core.layer import Layer
from .core.optim import *

import numpy as np


class NeuralNetwork:
    """ Our Neural Network container class.
    """
    def __init__(self, layers):
        self.layers = layers
        self.velocity = []
        
    def _loss(self, X, Y):
        Y_pred = self.predict(X)
        return self.layers[-1].loss(Y, Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        activity = X
        for layer in self.layers:
            activity = layer.fprop(activity)
        Y_pred = activity
        return Y_pred
    
    def backpropagate(self, Y, Y_pred, upto=0):
        """ Backpropagation of partial derivatives through 
            the complete network up to layer 'upto'
        """
        next_grad = self.layers[-1].input_grad(Y, Y_pred)
        for layer in self.layers[-2::-1]:
            next_grad = layer.bprop(next_grad)
        return next_grad

    def classification_error(self, X, Y):
        """ Calculate error on the given data 
            assuming they are classes that should be predicted. 
        """
        Y_pred = unhot(self.predict(X))
        error = Y_pred != Y
        return np.mean(error)

    def sgd_epoch(self, X, Y, learning_rate, momentum, batch_size):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        for b in range(n_batches):
            # start by extracting a batch from X and Y
            # (you can assume the inputs are already shuffled)
            inputs = X[batch_size*b:batch_size*(b + 1)]
            targets = Y[batch_size*b:batch_size*(b + 1)]
            # then forward and backward propagation + updates
            predictions = self.predict(inputs)
            self.backpropagate(targets, predictions)
            # HINT: layer.params() returns parameters *by reference*
            #       so you can easily update in-place
            if momentum is not None:
                i = 0
                for layer in self.layers:
                    if isinstance(layer, Parameterized):
                        paired = zip(layer.params(), layer.grad_params(), self.velocity[i])
                        for (param, grad, vel) in paired:
                            vel *= momentum
                            vel-= learning_rate * grad
                            param += vel

                        i += 1
            else:
                for layer in self.layers:
                    if isinstance(layer, Parameterized):
                        paired = zip(layer.params(), layer.grad_params())
                        for (param, grad) in paired:
                            param -= learning_rate * grad

    def gd_epoch(self, X, Y, learning_rate):
        n_samples = X.shape[0]
        self.sgd_epoch(X, Y, learning_rate, n_samples)

    def train(self, X, Y,
            X_valid=None, Y_valid=None,
            learning_rate=0.1, momentum=None,
            max_epochs=100, batch_size=64, descent_type="sgd", y_one_hot=True):
        for layer in self.layers:
            if isinstance(layer, Parameterized):
                params = layer.params()
                tup = params[0] * 0.0, params[1] * 0.0
                self.velocity.append(tup)

        """ Train network on the given data. """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        if y_one_hot:
            Y_train = one_hot(Y)
        else:
            Y_train = Y
        print("... starting training")
        for e in range(max_epochs+1):
            if descent_type == "sgd":
                self.sgd_epoch(X, Y_train, learning_rate, momentum, batch_size)
            elif descent_type == "gd":
                self.gd_epoch(X, Y_train, learning_rate)
            else:
                raise NotImplementedError("Unknown gradient descent type {}".format(descent_type))

            # Output error on the training data
            train_loss = self._loss(X, Y_train)
            train_error = self.classification_error(X, Y)
            # compute error on validation data:
            # simply make the function take validation data as input
            # and then compute errors here and print them
            if X_valid is not None:
                valid_error = self.classification_error(X_valid, Y_valid)

            print('epoch {:.4f}, loss {:.4f}, train error {:.4f}, validation error {:.4f}'.format(
                e, train_loss, train_error, valid_error))
    
    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        for l, layer in enumerate(self.layers):
            if isinstance(layer, Parameterized):
                print('checking gradient for layer {}'.format(l))
                for p, param in enumerate(layer.params()):
                    # we iterate through all parameters
                    param_shape = param.shape
                    # define functions for conveniently swapping
                    # out parameters of this specific layer and 
                    # computing loss and gradient with these 
                    # changed parametrs
                    def output_given_params(param_new):
                        """ A function that will compute the output 
                            of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # return computed loss
                        return self._loss(X, Y)

                    def grad_given_params(param_new):
                        """A function that will compute the gradient 
                           of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # Forward propagation through the net
                        Y_pred = self.predict(X)
                        # Backpropagation of partial derivatives
                        self.backpropagate(Y, Y_pred, upto=l)
                        # return the computed gradient 
                        return np.ravel(self.layers[l].grad_params()[p])

                    # let the initial parameters be the ones that
                    # are currently placed in the network and flatten them
                    # to a vector for convenient comparisons, printing etc.
                    param_init = np.ravel(np.copy(param))
                    
                    # TODO ####################################
                    # TODO compute the gradient with respect to
                    #      the initial parameters in two ways:
                    #      1) with grad_given_params()
                    #      2) with finite differences 
                    #         using output_given_params()
                    #         (as discussed in the lecture)
                    #      if your implementation is correct 
                    #      both results should be epsilon close
                    #      to each other!
                    # TODO ####################################
                    epsilon = 1e-4
                    # making sure your gradient checking routine itself 
                    # has no errors can be a bit tricky. To debug it
                    # you can "cheat" by using scipy which implements
                    # gradient checking exactly the way you should!
                    # To do that simply run the following here:
                    import scipy.optimize
                    #scipy_err = scipy.optimize.check_grad(output_given_params, 
                    #        grad_given_params, param_init, epsilon=epsilon)
                    #print('scipy err {:.2e}'.format(scipy_err))
                    loss_base = output_given_params(param_init)
                    # this should hold the gradient as calculated through bprop
                    gparam_bprop = grad_given_params(param_init)
                    # this should hold the gradient calculated through 
                    # finite differences
                    gparam_fd = np.zeros_like(param_init)
                    for i in range(len(param_init)):
                        param_init[i] += epsilon
                        gparam_fd[i] = (output_given_params(param_init) - loss_base) / epsilon
                        param_init[i] -= epsilon
                    # calculate difference between them
                    err = np.mean(np.abs(gparam_bprop - gparam_fd))
                    print('diff {:.2e}'.format(err))
                    assert(err < epsilon)
                    
                    # reset the parameters to their initial values
                    param[:] = np.reshape(param_init, param_shape)

